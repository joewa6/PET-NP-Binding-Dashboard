#!/usr/bin/env python3
"""
PET–Contaminant Binding Dashboard (Sustech Flagship)

Tabs:
1. Leaderboard – global binding rankings and top performers
2. System Details – single-system analysis with 3D viewer and atom hot-spots
3. QC – quality control, distributions, and diagnostics
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from io import BytesIO
try:
    import streamlit.components.v1 as components
except ImportError:
    components = None

# === PATHS ===============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent                  # .../PET_Contaminant
BASE_DIR = PROJECT_ROOT / "data"               # .../PET_Contaminant/data

# === DATA LOADER =================================================
@st.cache_data
def load_data_50(base_dir: Path):
    """Load pH 7.4 (−50 charge) data and convert time units to nanoseconds"""
    summ_dir = base_dir / "_50_" / "summaries"
    
    systems_csv = summ_dir / "summary_systems.csv"
    analytes_csv = summ_dir / "summary_analytes.csv"
    atoms_pq = summ_dir / "summary_atoms.parquet"
    
    # Check if files exist
    if not systems_csv.exists() or not analytes_csv.exists() or not atoms_pq.exists():
        st.error(f"Summaries not found under: {summ_dir}")
        st.stop()
    
    sys_df = pd.read_csv(systems_csv)
    an_df = pd.read_csv(analytes_csv)
    at_df = pd.read_parquet(atoms_pq)
    
    # Convert time units to nanoseconds
    if 'tau_ps_mean' in sys_df.columns:
        sys_df["tau_ns_mean"] = sys_df["tau_ps_mean"] / 1000
        sys_df["tau_ns_ci_lo"] = sys_df.get("tau_ps_ci_lo", 0) / 1000
        sys_df["tau_ns_ci_hi"] = sys_df.get("tau_ps_ci_hi", 0) / 1000
    if 'dt_ps' in sys_df.columns:
        sys_df["dt_ns"] = sys_df["dt_ps"] / 1000
    if 'tau_ps_mean' in an_df.columns:
        an_df["tau_ns_mean"] = an_df["tau_ps_mean"] / 1000
    
    return sys_df, an_df, at_df

# === VISUAL STYLE ========================================================
PALETTES = {
    "seq": px.colors.sequential.Teal,
    "seq_alt": px.colors.sequential.Blues,
    "div": px.colors.diverging.RdBu_r,
    "cat": px.colors.qualitative.Set2,
}

PLOTLY_COMMON = dict(
    template="simple_white",
    color_discrete_sequence=PALETTES["cat"],
)

def pct(x): 
    return 100.0 * x

# === MOLECULAR DATA PROCESSING =============================================
def guess_element(atype: str, aname: str) -> str:
    """
    Map SYBYL/Tripos atom types (antechamber) to element symbols.
    - Preserves halogens (Cl/Br/I/F) and common two-letter elements.
    - Maps aromatic/valence-labeled types (ca, c3, n2, oh, os, etc.) to base elements.
    """
    t = (atype or "").strip().lower()
    n = (aname or "").strip()

    # If atom name already looks like an element (e.g., "Cl1", "C5"), prefer it.
    if n:
        # Two-letter first (Cl, Br, Si, Na, Li, Ca, Fe, Al, Mg, Zn, Cu, Ni, Mn, Co, Hg, Pb, Sn, Cr, Ti)
        if len(n) >= 2 and n[:2].istitle() and n[:2] in {
            "Cl","Br","Si","Na","Li","Ca","Fe","Al","Mg","Zn",
            "Cu","Ni","Mn","Co","Hg","Pb","Sn","Cr","Ti"
        }:
            return n[:2]
        # Single-letter element in name
        if n[0].isalpha() and n[0].isupper():
            if n[0] in {"C","N","O","S","P","H","B","F","I","K"}:
                return n[0]

    # SYBYL/antechamber organic types → base element
    map_type = {
        # carbon
        "c":"C","ca":"C","c1":"C","c2":"C","c3":"C","cp":"C","cc":"C","cd":"C","ce":"C","cf":"C","cg":"C","ch":"C","c.cat":"C",
        # nitrogen
        "n":"N","n1":"N","n2":"N","n3":"N","n4":"N","na":"N","nb":"N","nc":"N","nd":"N","ne":"N","nh":"N","no":"N",
        # oxygen
        "o":"O","oh":"O","os":"O","op":"O","ox":"O",
        # sulfur
        "s":"S","sh":"S","so":"S","so2":"S","s2":"S","s.o":"S",
        # phosphorus
        "p":"P","pb":"P","pc":"P","pd":"P","pe":"P","px":"P",
        # hydrogen
        "h":"H","h1":"H","ha":"H","hn":"H","ho":"H","hs":"H",
        # halogens (also appear as raw element)
        "cl":"Cl","br":"Br","i":"I","f":"F",
        # silicon / boron
        "si":"Si","b":"B",
        # metals (rare here; fallback if encountered)
        "li":"Li","na_elem":"Na","k":"K","mg":"Mg","ca_elem":"Ca","zn":"Zn","cu":"Cu","ni":"Ni",
        "mn":"Mn","co":"Co","hg":"Hg","pb_elem":"Pb","sn":"Sn","cr":"Cr","ti":"Ti","al":"Al","fe":"Fe",
    }

    # Disambiguate tokens that collide with aromatic N/metal symbols:
    # 'na' in SYBYL is aromatic nitrogen (→ N), not sodium. We map above to N.
    # If true sodium appears, it often comes as atom name "Na", caught earlier.

    if t in map_type:
        return map_type[t]

    # Preserve two-letter element prefixes in type when present
    for two in ("cl","br","si","na","li","ca","fe","al","mg","zn","cu","ni","mn","co","hg","pb","sn","cr","ti"):
        if t.startswith(two):
            # aromatic N 'na' should be N; special-case
            if two == "na":  # likely aromatic N
                return "N"
            if two == "ca":  # aromatic carbon, not calcium
                return "C"
            return two.capitalize()

    # Fall back: first alphabetic char → element (C as last resort)
    for ch in t:
        if ch.isalpha():
            return ch.upper()
    return "C"

def clean_mol2_to_rdkit(path: Path):
    """Robust loader for antechamber mol2 -> RDKit Mol (preserves halogens; fixes ca→C, na→N, etc.)."""
    try:
        from rdkit import Chem
    except ImportError:
        return None
        
    mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    if mol is None:
        txt = path.read_text().splitlines()
        fixed, in_atom = [], False
        for line in txt:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                fixed.append(line)
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom = False
                fixed.append(line)
                continue
            if in_atom and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    aname = parts[1]
                    atype = parts[5]
                    elem = guess_element(atype, aname)
                    parts[5] = elem
                    line = " ".join(parts)
                fixed.append(line)
            else:
                fixed.append(line)
        block = "\n".join(fixed)
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol

def render_rdkit_with_mol2_labels(mol, size=(400, 400)):
    """
    Render a 2D RDKit molecule with each atom labeled using its MOL2 atom name.
    Trims whitespace and fills the image tightly around the molecule.
    Returns a BytesIO PNG buffer usable by st.image().
    """
    if mol is None:
        return None

    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image, ImageChops
        from io import BytesIO

        # Ensure 2D coordinates exist
        rdDepictor.Compute2DCoords(mol)

        # Create drawer with higher resolution for better quality (render large, display small)
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)  # High resolution canvas
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.fixedBondLength = 60      # larger bonds for high-res rendering
        opts.bondLineWidth = 3.0       # thicker for high-res rendering
        opts.padding = 0.0             # no whitespace

        # Assign MOL2 atom labels
        for atom in mol.GetAtoms():
            if atom.HasProp("_TriposAtomName"):
                atom_name = atom.GetProp("_TriposAtomName")
            else:
                atom_name = atom.GetSymbol() + str(atom.GetIdx())
            opts.atomLabels[atom.GetIdx()] = atom_name

        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert to PIL image
        png_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(png_data)).convert("RGBA")

        # Find the actual content and crop aggressively
        # Create a mask of non-white pixels
        white = (255, 255, 255, 255)
        pixels = img.load()
        width, height = img.size
        
        # Find bounds of non-white content
        left, top, right, bottom = width, height, 0, 0
        for y in range(height):
            for x in range(width):
                if pixels[x, y] != white:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
        
        # Add small margin and crop
        margin = 10
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(width, right + margin)
        bottom = min(height, bottom + margin)
        
        if left < right and top < bottom:
            img = img.crop((left, top, right, bottom))

        # Save to BytesIO buffer
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"[render_rdkit_with_mol2_labels] Failed: {e}")
        return None

# === 3D MOLECULAR VIEWER (REAL DATA + FALLBACK) =========================
def create_real_molecule_viewer(system_name, state_key="_50_"):
    """Create 3D viewer using actual molecular data (py3Dmol + RDKit)"""
    try:
        import py3Dmol
        from rdkit import Chem
        from collections import Counter
    except ImportError:
        return None, "py3Dmol and RDKit required for real molecular data", None
    
    # Look for molecular data in the appropriate state directory with correct naming convention
    # Both _35_ and _50_ now use the same structure: SystemName/LIG.mol2
    lig_path = BASE_DIR / state_key / system_name / "LIG.mol2"
    
    if not lig_path.exists():
        return None, f"No LIG.mol2 found for {system_name}", None
    
    try:
        mol = clean_mol2_to_rdkit(lig_path)
        if mol is None:
            raise ValueError("RDKit could not parse this MOL2 file.")

        mb = Chem.MolToMolBlock(mol)  # keep hydrogens
        view = py3Dmol.view(width='100%', height=420)
        view.addModel(mb, "mol")
        view.setStyle({"stick": {"colorscheme": "Jmol"}})
        view.setBackgroundColor("0xffffff")

        # Force bounding box recalc + centering (cross-browser safe)
        view.zoomTo({"model": -1})       # zoom to last added model
        view.center({"model": -1})       # force camera recenter
        view.zoom(1.0)                   # small zoom-out
        view.render()

        # JS fallback for late WebGL init (Safari fix)
        view.addScript("""
        setTimeout(function(){
            try {
                viewer.center({model: -1});
                viewer.zoomTo({model: -1});
                viewer.render();
            } catch(e) { console.error('Recenter failed:', e); }
        }, 500);
        """)

        # === COMPACT LEGEND OVERLAY ===========================================
        jmol_colors = {
            "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
            "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
            "S": "#FFFF30", "P": "#FF8000", "Si": "#F0C8A0", "B": "#FFB5B5",
            "Na": "#AB5CF2", "K": "#8F40D4", "Ca": "#3DFF00",
        }
        
        elems = [a.GetSymbol() for a in mol.GetAtoms()]
        counts = Counter(elems)
        order = sorted(counts.items(), key=lambda kv: (kv[0] in {"H"}, kv[0]))
        
        legend_items = []
        for sym, n in order:
            color = jmol_colors.get(sym, "#909090")
            legend_items.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<span style='width:10px;height:10px;background:{color};"
                f"border:1px solid #666;border-radius:2px;display:inline-block;margin-right:6px;'></span>"
                f"<span style='font-size:12px;'>{sym}<span style='color:#666;'>&nbsp;×{n}</span></span>"
                f"</div>"
            )

        # Generate 2D sketch with MOL2 atom labels
        mol2d_img = render_rdkit_with_mol2_labels(mol)
        
        # Get the base HTML - no complex injection needed
        view_html = view._make_html()
        
        return view_html, mol, mol2d_img
        
    except Exception as e:
        return None, f"Failed to render molecule: {e}", None

def create_3dmol_viewer(system_name):
    """Create a 3D molecular viewer using 3Dmol.js (fallback)"""
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <style>
            .viewer_3Dmoljs {{
                height: 400px; 
                width: 100%; 
                position: relative;
                border: 2px solid #e0fbfc;
                border-radius: 8px;
                background-color: #ffffff;
            }}
            .viewer-title {{
                text-align: center; 
                margin-top: 10px; 
                color: #0a2540; 
                font-size: 14px;
                font-weight: 600;
            }}
            .loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #005f73;
                font-size: 16px;
            }}
        </style>
    </head>
    <body>
        <div id="molviewer" class="viewer_3Dmoljs">
            <div class="loading">🧬 Loading molecular structure...</div>
        </div>
        <div class="viewer-title">Interactive 3D view of {system_name}</div>
        
        <script>
            console.log('Starting 3D molecular viewer initialization...');
            
            let viewer = null;
            let isInitialized = false;
            
            function initViewer() {{
                try {{
                    console.log('3Dmol check:', typeof $3Dmol !== 'undefined' ? 'loaded' : 'not loaded');
                    
                    if (typeof $3Dmol === 'undefined') {{
                        document.getElementById('molviewer').innerHTML = '<div style="padding: 20px; text-align: center; color: #ca6702;">⚠️ 3Dmol.js library not loaded</div>';
                        return;
                    }}
                    
                    let element = document.getElementById('molviewer');
                    element.innerHTML = ''; // Clear loading message
                    
                    let config = {{ 
                        backgroundColor: 'white',
                        antialias: true
                    }};
                    
                    viewer = $3Dmol.createViewer(element, config);
                    console.log('Viewer created successfully');
                    
                    // Enhanced PET fragment with contaminant
                    let pdb_data = `HEADER    {system_name.replace(' ', '_').upper()}_BINDING_SITE
REMARK    PET surface (chain A) with bound contaminant (chain B)
ATOM      1  C   PET A   1      -3.445   1.663   0.000  1.00 20.00           C
ATOM      2  C   PET A   1      -2.222   2.331   0.000  1.00 20.00           C
ATOM      3  C   PET A   1      -1.000   1.663   0.000  1.00 20.00           C
ATOM      4  C   PET A   1      -1.000   0.272   0.000  1.00 20.00           C
ATOM      5  C   PET A   1      -2.222  -0.396   0.000  1.00 20.00           C
ATOM      6  C   PET A   1      -3.445   0.272   0.000  1.00 20.00           C
ATOM      7  O   PET A   1       0.222   2.331   0.000  1.00 20.00           O
ATOM      8  O   PET A   1       0.222  -0.396   0.000  1.00 20.00           O
ATOM      9  C   PET A   1       1.445   1.663   0.000  1.00 20.00           C
ATOM     10  O   PET A   1       2.667   2.331   0.000  1.00 20.00           O
ATOM     11  C   PET A   1      -3.445   1.663  -2.000  1.00 20.00           C
ATOM     12  C   PET A   1      -2.222   2.331  -2.000  1.00 20.00           C
ATOM     13  C   CON B   2       2.000   0.500   1.500  1.00 50.00           C
ATOM     14  N   CON B   2       3.200   0.800   1.800  1.00 50.00           N
ATOM     15  O   CON B   2       1.500  -0.300   2.200  1.00 50.00           O
ATOM     16  C   CON B   2       3.800   1.600   1.200  1.00 50.00           C
ATOM     17  C   CON B   2       0.800   0.000   0.800  1.00 50.00           C
ATOM     18  H   CON B   2       4.500   1.800   1.500  1.00 50.00           H
END`;
                    
                    viewer.addModel(pdb_data, "pdb");
                    console.log('Model added to viewer');
                    
                    // Style PET surface (blue theme)
                    viewer.setStyle({{'chain': 'A'}}, {{
                        stick: {{radius: 0.12, color: '#005f73'}}, 
                        sphere: {{scale: 0.25, color: '#005f73'}}
                    }});
                    
                    // Style contaminant (orange theme)
                    viewer.setStyle({{'chain': 'B'}}, {{
                        stick: {{radius: 0.15, color: '#ca6702'}}, 
                        sphere: {{scale: 0.35, color: '#ca6702'}}
                    }});
                    
                    // Add subtle surface for PET only
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.2, 
                        color: '#e0fbfc'
                    }}, {{'chain': 'A'}});
                    
                    // Position camera and render
                    viewer.zoomTo();
                    viewer.render();
                    console.log('Initial render complete');
                    
                    // Fix for 3Dmol viewer centering - delayed re-center for browser compatibility
                    setTimeout(function() {{
                        viewer.zoomTo();
                        console.log('Delayed zoomTo applied for centering fix');
                    }}, 500);
                    
                    // Add click interaction
                    viewer.setClickable({{}}, true, function(atom, viewer, event, container) {{
                        console.log('Atom clicked:', atom);
                        if (atom && atom.chain && atom.resi && atom.atom) {{
                            viewer.addLabel(atom.chain + ':' + atom.resi + ':' + atom.atom, {{
                                position: atom, 
                                backgroundColor: 'rgba(0,0,0,0.8)', 
                                fontColor: 'white',
                                fontSize: 12
                            }});
                            viewer.render();
                        }}
                    }});
                    
                    isInitialized = true;
                    console.log('3D viewer fully initialized');
                    
                    // --- Robust recenter on resize or tab change ---
                    let resizeTimeout = null;
                    window.addEventListener('resize', function() {{
                        clearTimeout(resizeTimeout);
                        resizeTimeout = setTimeout(function() {{
                            if (viewer && isInitialized) {{
                                try {{
                                    viewer.resize();
                                    viewer.zoomTo({{model: -1}});
                                    viewer.render();
                                    console.log('Resize re-center applied');
                                }} catch (e) {{
                                    console.error('Resize re-center failed:', e);
                                }}
                            }}
                        }}, 200);
                    }});
                    
                    // Also recenter when tab becomes visible again (Streamlit tab fix)
                    document.addEventListener('visibilitychange', function() {{
                        if (!document.hidden && viewer && isInitialized) {{
                            try {{
                                viewer.center({{model: -1}});
                                viewer.zoomTo({{model: -1}});
                                viewer.render();
                                console.log('Visibility re-center applied');
                            }} catch (e) {{
                                console.error('Visibility re-center failed:', e);
                            }}
                        }}
                    }});
                    
                }} catch (error) {{
                    console.error('Error initializing 3D viewer:', error);
                    document.getElementById('molviewer').innerHTML = '<div style="padding: 20px; text-align: center; color: #ca6702;">❌ Error loading 3D structure: ' + error.message + '</div>';
                }}
            }}
            
            // Manual rotation function (user can trigger)
            function rotateViewer() {{
                if (viewer && isInitialized) {{
                    viewer.rotate(10, 'y');
                    viewer.render();
                }}
            }}
            
            // Multiple initialization attempts
            function tryInit() {{
                if (typeof $3Dmol !== 'undefined') {{
                    initViewer();
                }} else {{
                    console.log('3Dmol not ready, retrying...');
                    setTimeout(tryInit, 200);
                }}
            }}
            
            // Start initialization
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(tryInit, 300);
                }});
            }} else {{
                setTimeout(tryInit, 300);
            }}
            
            // Add manual controls
            setTimeout(function() {{
                if (isInitialized) {{
                    let controls = document.createElement('div');
                    controls.innerHTML = '<button onclick="rotateViewer()" style="margin-top: 5px; padding: 5px 10px; background: #005f73; color: white; border: none; border-radius: 4px; cursor: pointer;">🔄 Rotate</button>';
                    document.querySelector('.viewer-title').appendChild(controls);
                }}
            }}, 2000);
            
        </script>
    </body>
    </html>
    """
    return html_string

# === TAB 1: LEADERBOARD =================================================
def leaderboard_tab(sys_df: pd.DataFrame):
    with st.container():
        st.markdown("### Global Binding Rankings")
        
        if sys_df.empty:
            st.info("No systems found.")
            return

        # Control panel
        col1, col2 = st.columns([2, 1])
        with col1:
            metric = st.radio("Rank by", ["Binding %", "Tau (ns)"], horizontal=True)
        with col2:
            max_n = int(len(sys_df))
            default_n = max_n  # Default to showing all systems
            top_n = st.slider("Show top N", 1, max_n, default_n, key="leader_topn")

    st.markdown("---")

    with st.container():
        st.markdown("#### Top Performing Systems")
        
        df = sys_df.copy()
        if metric == "Binding %":
            df["metric"] = pct(df["binding_frac_mean"])
            df["ci_lo"] = pct(df["binding_frac_ci_lo"])
            df["ci_hi"] = pct(df["binding_frac_ci_hi"])
            ylab = "Binding %"
        else:
            df["metric"] = df["tau_ns_mean"]  # Already converted to ns
            df["ci_lo"] = df["tau_ns_ci_lo"]
            df["ci_hi"] = df["tau_ns_ci_hi"]
            ylab = "Tau (ns)"

        df = df.sort_values("metric", ascending=False).head(top_n)

        fig = px.bar(df, x="system", y="metric", 
                     color="metric",
                     color_continuous_scale=PALETTES["seq"],
                     hover_data=["n_replicas", "dt_ps"],
                     labels={"metric": ylab, "system": "System"},
                     **PLOTLY_COMMON)
        
        # Add confidence intervals
        for _, row in df.iterrows():
            if np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"]):
                fig.add_shape(
                    type="line",
                    x0=row["system"], x1=row["system"],
                    y0=row["ci_lo"], y1=row["ci_hi"],
                    line=dict(width=2, color="rgba(0,0,0,0.6)"),
                )
        
        fig.update_layout(
            xaxis_tickangle=-45, height=600, margin=dict(l=10, r=10, t=30, b=120),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=14, color="#0a2540"),
            coloraxis_colorbar=dict(outlinewidth=0, ticks="outside"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Top {top_n} systems ranked by {metric.lower()} with 95% confidence intervals.")

        with st.expander("📊 Detailed Data Table"):
            show_cols = [
                "system",
                "binding_frac_mean", "binding_frac_ci_lo", "binding_frac_ci_hi",
                "tau_ps_mean", "tau_ps_ci_lo", "tau_ps_ci_hi",
                "dt_ps", "n_replicas", "frames_min", "frames_max",
                "analytes_used", "analytes_total",
            ]
            st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True)
            
            # Download button for leaderboard data
            csv_data = df[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Leaderboard Data (CSV)",
                data=csv_data,
                file_name="leaderboard_top_systems.csv",
                mime="text/csv",
            )

# === TAB 2: SYSTEM DETAILS ==============================================
def system_tab(sys_df: pd.DataFrame, an_df: pd.DataFrame, at_df: pd.DataFrame, state_key: str = "_50_"):
    with st.container():
        st.markdown("### Single-System Analysis")

        systems = sys_df.sort_values("binding_frac_mean", ascending=False)["system"].tolist()
        if not systems:
            st.info("No systems available.")
            return

        sel = st.selectbox("Select system for detailed analysis", systems)

    st.markdown("---")

    # Get system metadata
    meta = sys_df.loc[sys_df["system"] == sel].iloc[0].to_dict()

    # === COMPACT STRUCTURE + METRICS LAYOUT =====================================
    with st.container():
        st.markdown("#### 3D Molecular Structure & System Overview")

        # Four-column layout: legend | 3D viewer | compact metrics | larger 2D image
        col_legend, col_3d, col_metrics, col_2d = st.columns([0.3, 1.5, 0.7, 1.2], gap="small")

        # --- LEFT: Bigger Legend ----------------------------------------
        with col_legend:
            view_html, mol, mol2d_img = create_real_molecule_viewer(sel, state_key)
            
            if view_html is not None and mol is not None:
                from collections import Counter
                jmol_colors = {
                    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
                    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
                    "S": "#FFFF30", "P": "#FF8000", "Si": "#F0C8A0", "B": "#FFB5B5"
                }
                elems = [a.GetSymbol() for a in mol.GetAtoms()]
                counts = Counter(elems)
                
                st.markdown("**Atom Colours**")
                for sym, n in sorted(counts.items(), key=lambda kv: (kv[0] in {'H'}, kv[0])):
                    color = jmol_colors.get(sym, "#ccc")
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin:4px 0;font-size:14px;'>"
                        f"<div style='width:16px;height:16px;background:{color};border:1px solid #666;margin-right:8px;'></div>"
                        f"<span><strong>{sym}</strong> <span style='color:#666;'>×{n}</span></span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # --- CENTER: 3D viewer --------------------------------------------
        with col_3d:
            if view_html is not None and components is not None:
                components.html(view_html, height=500, scrolling=False)
            else:
                st.warning("3D view unavailable")

        # --- METRICS: Single column with key metrics ----------------------
        with col_metrics:
            st.markdown("**📈 Key Metrics**")
            
            # Calculate ranks for binding and tau
            binding_rank = (sys_df["binding_frac_mean"] >= meta["binding_frac_mean"]).sum()
            tau_rank = (sys_df["tau_ns_mean"] >= meta["tau_ns_mean"]).sum()
            
            st.metric(
                "Binding %", 
                f"{pct(meta['binding_frac_mean']):.1f}",
                delta=f"Rank {binding_rank}"
            )
            st.metric(
                "τ (ns)", 
                f"{meta['tau_ns_mean']:.2f}",
                delta=f"Rank {tau_rank}"
            )
            st.caption(
                f"Δt = {meta['dt_ns']:.3f} ns • Reps = {int(meta['n_replicas'])}"
            )

        # --- RIGHT: Larger 2D molecular structure -------------------------
        with col_2d:
            # Much larger 2D sketch - high quality, more space for big molecules!
            if mol2d_img:
                st.markdown("**2D Structure**")
                st.markdown(
                    "<div style='display: flex; justify-content: center; align-items: center; margin: 0.5rem 0;'>", 
                    unsafe_allow_html=True
                )
                st.image(mol2d_img, width=240)  # Fixed width for balanced appearance across screen sizes
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("2D structure unavailable")

    st.markdown("---")

    # Emphasize atomic hotspot analysis (now the main analytical block)
    with st.container():
        st.markdown("#### 🎯 Atomic Hot-Spot Analysis")
        st.markdown("**Primary Analytical Focus: Atomic-Level Binding Interactions**")
        st.caption("P(atom bound) during binding events - identifying key interaction sites")

        # Unified per-system atom probabilities
        atoms = at_df[at_df["system"] == sel].copy()
        if atoms.empty:
            st.info("No atom data for this system.")
            return

        atoms = (
            atoms.groupby("atom", as_index=False)
            .agg({"p_bound": "mean"})
            .sort_values("p_bound", ascending=False)
        )
        atoms["P%"] = pct(atoms["p_bound"])

        # Add element information for easier interpretation
        atoms["Element"] = atoms["atom"].str.extract(r'^([A-Z][a-z]?)')

        max_atoms = int(min(200, len(atoms)))
        if max_atoms <= 0:
            st.info("No atoms to display.")
            return
        default_atoms = min(50, max_atoms)
        top_n = st.slider("Top N atoms", 1, max_atoms, default_atoms, key="atoms_topn")

        atoms_top = atoms.head(top_n)

        fig3 = px.bar(atoms_top, x="atom", y="P%", 
                      color="P%",
                      color_continuous_scale=PALETTES["seq"],
                      hover_data=["p_bound", "Element"],
                      labels={"atom": "Atom (MOL2 Name)", "P%": "P(bound) %", "Element": "Element"},
                      template="simple_white",
                      range_color=[0, 100])  # Fix color scale to 0-100%
        fig3.update_layout(
            xaxis_tickangle=-45, height=500, margin=dict(l=10, r=10, t=30, b=100),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=14, color="#0a2540"),
            coloraxis_colorbar=dict(
                title="P(bound) %",
                outlinewidth=0, 
                ticks="outside",
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]
            ),
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Atomic-level binding probabilities revealing key interaction sites. Color intensity indicates binding strength.")

        with st.expander("📋 Complete Atom Data Table"):
            st.dataframe(atoms[["atom", "Element", "p_bound", "P%"]].reset_index(drop=True), use_container_width=True)

        # Download button for atom data
        st.download_button(
            "📥 Download Atom Hot-Spot Data (CSV)",
            data=atoms.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_atoms.csv",
            mime="text/csv",
        )

# === COMPARISON TAB =====================================================
# === TAB 3: GENERAL TRENDS ==============================================
def qc_tab(sys_df: pd.DataFrame):
    with st.container():
        st.markdown("### General Trends")
        
        if sys_df.empty:
            st.info("No systems found.")
            return

    st.markdown("---")

    with st.container():
        st.markdown("#### Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Systems", len(sys_df))
        # Remove the other metrics as requested
        c2.write("")  # Empty column
        c3.write("")  # Empty column

    st.markdown("---")

    with st.container():
        st.markdown("#### τ (ns) vs Binding % Correlation")
        st.caption("Explore the relationship between residence time and binding efficiency across all systems")
        
        # Prepare data for scatter plot - use already converted ns values
        scatter_df = sys_df.copy()
        scatter_df["binding_pct"] = pct(scatter_df["binding_frac_mean"])  # Convert to percentage
        
        # Calculate slope for trendline through origin (0,0)
        # For y = mx (no intercept), optimal slope m = sum(xy) / sum(x^2)
        x_vals = scatter_df["binding_pct"].values
        y_vals = scatter_df["tau_ns_mean"].values
        slope = np.sum(x_vals * y_vals) / np.sum(x_vals ** 2)
        
        # Create scatter plot
        fig_scatter = px.scatter(
            scatter_df, 
            x="binding_pct", 
            y="tau_ns_mean",  # Use pre-converted nanosecond values
            hover_data=["system", "binding_frac_mean", "tau_ns_mean"],
            labels={
                "binding_pct": "Binding %", 
                "tau_ns_mean": "τ (ns)",
                "system": "System"
            },
            title="Residence Time vs Binding Efficiency",
            **PLOTLY_COMMON
        )
        
        # Add manual trendline through origin
        x_range = np.array([0, scatter_df["binding_pct"].max()])
        y_fit = slope * x_range
        fig_scatter.add_scatter(
            x=x_range, 
            y=y_fit, 
            mode='lines',
            name=f'Fit (slope={slope:.4f})',
            line=dict(color='rgba(255,0,0,0.6)', width=2, dash='dash'),
            showlegend=True
        )
        
        # Update hover template for better information
        fig_scatter.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Binding: %{x:.1f}%<br>" +
                         "τ: %{y:.3f} ns<br>" +
                         "<extra></extra>",
            customdata=scatter_df[["system", "binding_frac_mean", "tau_ns_mean"]].values,
            selector=dict(mode='markers')
        )
        
        fig_scatter.update_layout(
            height=500, 
            margin=dict(l=10, r=10, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=14, color="#0a2540"),
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, None])
        fig_scatter.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, None])
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Each point represents a system. Color intensity indicates binding efficiency. Hover for detailed system information.")

    st.markdown("---")

    with st.container():
        st.markdown("#### Binding % Distribution")
        
        fig = px.histogram(100 * sys_df["binding_frac_mean"], nbins=30,
                          labels={"value": "Binding %", "count": "Number of Systems"}, 
                          title="Binding % Distribution",
                          **PLOTLY_COMMON)
        # Override color for histogram specifically
        fig.update_traces(marker_color=PALETTES["seq"][5])
        fig.update_layout(
            height=400, margin=dict(l=10, r=10, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=14, color="#0a2540"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribution of binding percentages across all systems for quality assessment.")

        with st.expander("📊 Complete System Table"):
            st.dataframe(sys_df.sort_values("system").reset_index(drop=True), use_container_width=True)
            
        # Download full dataset
        csv_data = sys_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Complete Dataset (CSV)",
            data=csv_data,
            file_name="complete_binding_analysis.csv",
            mime="text/csv",
        )

    # === Saved QC plots from ANALYSIS/PLOTS ==================================
    with st.container():
        st.markdown("#### Saved Analysis Plots")
        plots_dir = BASE_DIR / "ANALYSIS" / "PLOTS"
        if not plots_dir.exists():
            st.info(f"No saved QC plots found at: {plots_dir}")
        else:
            # Find specific plots we want to display
            correlation_heatmap = plots_dir / "correlation_heatmap.png"
            descriptor_scatter = plots_dir / "descriptor_scatter_plots.png"
            
            if correlation_heatmap.exists() or descriptor_scatter.exists():
                # Create 2-column layout: correlation matrix (much smaller) and descriptor scatter (much wider)
                col1, col2 = st.columns([1, 3], gap="medium")
                
                with col1:
                    if correlation_heatmap.exists():
                        st.markdown("**Correlation Matrix**")
                        try:
                            st.image(str(correlation_heatmap), use_container_width=True)
                        except Exception as e:
                            st.error(f"Unable to display correlation heatmap: {e}")
                    else:
                        st.info("Correlation heatmap not found")
                
                with col2:
                    if descriptor_scatter.exists():
                        st.markdown("**Descriptor Relationships**") 
                        try:
                            st.image(str(descriptor_scatter), use_container_width=True)
                        except Exception as e:
                            st.error(f"Unable to display descriptor scatter plots: {e}")
                    else:
                        st.info("Descriptor scatter plots not found")
            else:
                st.info("No correlation heatmap or descriptor scatter plots found in ANALYSIS/PLOTS.")

# === MAIN ================================================================
def main():
    st.set_page_config(page_title="PET–Contaminant Binding Dashboard", layout="wide")
    
    # === PROFESSIONAL STYLING ===============================================
    st.markdown(
        """
        <style>
        /* Global font + colour tweaks */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #1a1a1a;
            background-color: #f9f9f9;
        }

        /* Headings */
        h1, h2, h3 {
            color: #0a2540;
            font-weight: 600;
        }

        /* Fix title positioning */
        h1 {
            margin-top: 2rem !important;
        }

        /* Streamlit metric cards */
        div[data-testid="stMetricValue"] {
            color: #005f73;
            font-weight: 700;
        }

        /* Plots alignment */
        .element-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }

        /* Sidebar and tabs */
        div[data-baseweb="tab"] {
            font-weight: 600;
        }

        /* Remove unnecessary Streamlit padding */
        section.main > div {
            padding-top: 2rem;
        }

        /* Softer card look for containers */
        .block-container {
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
            background: #ffffff;
            padding: 1.2rem 2rem;
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
        }

        /* Reduce vertical spacing between Streamlit elements */
        .element-container {
            margin-top: 0rem !important;
            margin-bottom: 0.2rem !important;
        }

        /* Compact expander spacing */
        details[open] > summary {
            margin-bottom: 0.3rem !important;
        }

        /* Accent colour for expander headers */
        .streamlit-expanderHeader {
            background-color: #f0f9ff;
            color: #0a2540;
            font-weight: 600;
        }

        /* Subtle hover on buttons */
        .stButton>button:hover {
            background-color: #0a2540 !important;
            color: #ffffff !important;
            transition: 0.2s;
        }

        /* Professional tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            border: 1px solid #e0e0e0;
            border-bottom: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-color: #005f73;
            border-bottom: 2px solid #005f73;
        }

        /* Responsive design for tablets and smaller screens */
        @media (max-width: 900px) {
            .block-container {
                padding: 0.8rem 1rem !important;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 13px;
                padding-left: 12px;
                padding-right: 12px;
            }
            .element-container {
                margin-bottom: 0.6rem !important;
            }
        }

        /* Mobile-specific optimizations */
        @media (max-width: 600px) {
            .stTabs [data-baseweb="tab"] {
                font-size: 12px;
            }
            h1, h2, h3 {
                font-size: 1.1rem;
            }
            .stImage img {
                max-width: 100% !important;
                height: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("PET Binding Dashboard — pH 7.4 (−50 charge)")
    
    # === PROJECT BANNER =====================================================
    st.markdown(
        """
        <div style='padding: 0.8rem; background-color: #e0fbfc; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #ca6702;'>
            <b>Sustech Flagship — Work Package: Nanoplastics Fate</b><br>
            <small>200+ contaminants | PET nanoparticle at pH 7.4 (−50 total charge)</small><br>
            <span style='color: #ca6702; font-weight: 600;'>🔬 Current analysis: pH 7.4 dataset</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load data
    sys_df, an_df, at_df = load_data_50(BASE_DIR)

    # === TABS ===============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Leaderboard", 
        "🔬 System Details", 
        "📊 General Trends",
        "🧪 Experiment & Validation",
        "📖 Walkthrough"
    ])
    
    with tab1:
        st.markdown("**Current dataset:** pH 7.4 (PET NP with −50 total charge)")
        leaderboard_tab(sys_df)
    
    with tab2:
        st.markdown("**Current dataset:** pH 7.4 (PET NP with −50 total charge)")
        system_tab(sys_df, an_df, at_df, state_key="_50_")
    
    with tab3:
        st.markdown("**Current dataset:** pH 7.4 (PET NP with −50 total charge)")
        qc_tab(sys_df)
    
    with tab4:
        st.markdown("### 🔬 Experiment–Simulation Integration")
        st.info("""
        This tab is designed for **experimental collaboration and validation**.

        **Focus areas for validation:**\n
        • Binding affinity measurements and adsorption isotherms. \n
        • Aggregation behavior and particle interactions. \n
        • Environmental conditions effects (pH, ionic strength, temperature). \n

        **Simulation predictions available for comparison:** \n
        - Binding fractions and residence times for 200+ contaminants. \n
        - Molecular-level interaction hotspots and mechanisms. \n
        - Relative binding rankings across chemical classes. \n

        **Collaborative opportunities:**
        - Validate computational predictions with experimental data
        - Identify discrepancies for model refinement
        - Guide experimental design using simulation insights
        """)


    with tab5:  
        # Step-by-step guide
        with st.container():
            st.markdown("#### 🚀 Getting Started")
            st.markdown("""
            This dashboard analyzes **molecular dynamics simulations** of 200+ contaminants binding to PET nanoparticles at pH 7.4.
            Each tab provides different insights into the binding behavior.
            """)
            
            st.markdown("#### 📋 Tab Overview")
            
            # Leaderboard explanation
            with st.expander("🏆 **Leaderboard** - Find the strongest binders"):
                st.markdown("""
                **What it shows:** Global rankings of all contaminant-PET systems
                
                **Key metrics:**
                - **Binding %**: Percentage of simulation time the contaminant spends bound to PET
                - **Tau (ns)**: Average residence time when bound
                
                **How to use:**
                1. Choose ranking metric: "Binding %" or "Tau (ns)"
                2. Adjust slider to show top N systems
                3. Hover over bars for detailed information
                4. Download data using the CSV button
                
                **Note:** High binding % indicates strong affinity; high tau indicates stable binding once formed.
                """)
            
            # System Details explanation  
            with st.expander("🔬 **System Details** - Deep dive into individual molecules"):
                st.markdown("""
                **What it shows:** Detailed analysis of any single contaminant-PET system
                
                **Features:**
                - **3D molecular structure** with interactive rotation
                - **2D chemical structure** with numbered atom labels
                - **Atomic hot-spots** showing which specific atoms bind most frequently
                - **Element legend** with Jmol color coding
                
                **How to use:**
                1. Select a system from the dropdown (sorted by binding strength)
                2. Explore the 3D structure by rotating/zooming
                3. **Key connection**: Use the 2D structure to identify atom numbers, then cross-reference with the atomic hot-spots chart
                4. Find high-probability atoms in the hot-spots chart and locate them by number in the 2D structure
                5. This reveals which specific atoms are most involved in binding interactions
                
                **Note:** The 2D atom numbers directly correspond to the atomic hot-spots data - use them together to pinpoint binding sites!
                """)
            
            # General Trends explanation
            with st.expander("📊 **General Trends** - Population-level patterns"):
                st.markdown("""
                **What it shows:** Statistical analysis across all systems
                
                **Key visualizations:**
                - **Correlation plot**: Binding % vs residence time relationship
                - **Distribution histogram**: Spread of binding percentages
                - **Analysis plots**: Molecular descriptor correlations (if available)
                
                **How to use:**
                1. Examine the correlation plot to understand binding vs stability trade-offs
                2. Check the histogram to see the distribution of binding strengths
                3. View correlation matrices to identify molecular features that drive binding
                
                **Note:** Strong correlation between binding % and tau suggests stable binding systems.
                """)
            
            # Experimental validation explanation
            with st.expander("🧪 **Experiment & Validation** - Future integration"):
                st.markdown("""
                **What it will show:** Comparison of simulation predictions with experimental data
                
                **Planned features:**
                - Experimental binding measurements
                - Validation of simulation accuracy
                - Outlier identification for model improvement
                
                **Status:** Currently in development - will be populated with experimental results.
                """)
        
        st.markdown("---")
        
        with st.container():
            st.markdown("#### 💡 Helpful Notes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Data Interpretation:**
                - Higher binding % = stronger affinity
                - Higher tau = more stable when bound
                - Check both metrics for complete picture
                - Use confidence intervals to assess reliability
                """)
            
            with col2:
                st.markdown("""
                **Navigation Notes:**
                - Start with Leaderboard for overview
                - Use System Details for molecular insights  
                - Check General Trends for patterns
                - Download data for external analysis
                """)
        
        st.markdown("---")
        
        with st.container():
            st.markdown("#### ❓ Troubleshooting")
            
            with st.expander("Common Issues & Solutions"):
                st.markdown("""
                **3D viewer not loading:**
                - Check internet connection (requires external libraries)
                - Try refreshing the page
                - Use 2D structure as alternative
                
                **Plots not displaying:**
                - Verify ANALYSIS/PLOTS directory exists
                - Check file permissions
                - Contact support if persistent
                
                **Data seems incomplete:**
                - This dashboard shows pH 7.4 data only
                - Additional pH conditions may be available separately
                - Check dataset documentation for coverage
                
                **Performance issues:**
                - Reduce number of systems shown in Leaderboard
                - Close unused browser tabs
                - Use download feature for large datasets
                """)
        
        st.markdown("---")
        
        st.info("""
        **Need more help?** Contact me at joesph.wallace@iit.it for assistance or further information.
        This dashboard is part of the Sustech Flagship project studying nanoplastics fate and transport.
        """)

    st.markdown("---")
    st.markdown(
        """
        <hr>
        <center>
        <small style='color: #6c757d;'>
        Developed by J. Wallace, IIT Genova — Sustech Flagship 2025
        </small>
        </center>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
