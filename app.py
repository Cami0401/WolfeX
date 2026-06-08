import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="WolfeX · Optimizador",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #07090f; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.topbar {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-logo {
    font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-chip {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8; border-radius: 8px; padding: 6px 14px;
    font-size: 0.8rem; font-weight: 500; margin-left: 6px;
}
.nav-chip.active { background: rgba(139,92,246,0.2); border-color: rgba(139,92,246,0.4); color: #a78bfa; }

.tut-step {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 10px;
    display: flex; gap: 14px; align-items: flex-start;
}
.tut-num {
    background: linear-gradient(135deg,#7c3aed,#6366f1);
    color: white; border-radius: 50%; width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
}
.tut-title { font-weight: 600; color: #e2e8f0; font-size: 0.95rem; margin-bottom: 4px; }
.tut-desc  { color: #64748b; font-size: 0.83rem; line-height: 1.6; }
.tut-code  {
    display: inline-block; background: rgba(167,139,250,0.1);
    color: #a78bfa; border: 1px solid rgba(167,139,250,0.2);
    border-radius: 6px; padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
}

.explain-box { border-radius: 14px; padding: 1.3rem 1.5rem; margin: 8px 0; border-left: 4px solid; }
.explain-box.ok   { background: rgba(52,211,153,0.07); border-color: #34d399; }
.explain-box.warn { background: rgba(251,191,36,0.07);  border-color: #fbbf24; }
.explain-box.info { background: rgba(96,165,250,0.07);  border-color: #60a5fa; }
.explain-title { font-weight: 700; font-size: 1rem; margin-bottom: 6px; }
.explain-box.ok   .explain-title { color: #34d399; }
.explain-box.warn .explain-title { color: #fbbf24; }
.explain-box.info .explain-title { color: #60a5fa; }
.explain-body { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; }
.explain-body strong { color: #cbd5e1; }

.hist-card {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 6px;
    cursor: pointer; transition: border-color 0.2s;
}
.hist-card:hover { border-color: rgba(139,92,246,0.4); }
.hist-func { font-family: 'JetBrains Mono',monospace; color: #a78bfa; font-size: 0.82rem; }
.hist-meta { color: #475569; font-size: 0.72rem; margin-top: 3px; }

section[data-testid="stSidebar"] { display: none; }
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MATEMÁTICA
# ──────────────────────────────────────────────
def fmt_expr(s): return s.replace("**","^")

def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";",",").split(",") if v.strip()]
        if len(vals) != n: raise ValueError(f"Se necesitan {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto inválido: {e}")

def build_functions(expr_text, n):
    expr_text = expr_text.replace("^","**")
    syms = sp.symbols(f"x1:{n+1}")
    ld = {f"x{i+1}": syms[i] for i in range(n)}
    ld.update({"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,
               "ln":sp.log,"sqrt":sp.sqrt,"pi":sp.pi,"e":sp.E,"E":sp.E,
               "abs":sp.Abs,"Abs":sp.Abs,"sinh":sp.sinh,"cosh":sp.cosh,
               "tanh":sp.tanh,"atan":sp.atan,"asin":sp.asin,"acos":sp.acos})
    try:
        expr = sp.sympify(expr_text, locals=ld)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar: {e}")
    unk = expr.free_symbols - set(syms)
    if unk: raise ValueError(f"Variables no permitidas: {', '.join(str(s) for s in unk)}.")
    ge = [sp.diff(expr,v) for v in syms]
    he = sp.hessian(expr, syms)
    fl = sp.lambdify((syms,), expr, "numpy")
    gl = sp.lambdify((syms,), ge, "numpy")
    hl = sp.lambdify((syms,), he, "numpy")
    def f(x):
        v = float(np.asarray(fl(tuple(x)), dtype=float))
        if not np.isfinite(v): raise FloatingPointError("f no finita")
        return v
    def g(x):
        gv = np.array(gl(tuple(x)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(gv)): raise FloatingPointError("grad no finito")
        return gv
    def h(x): return np.array(hl(tuple(x)), dtype=float)
    return expr, f, g, h

def wolfe_ls(f, g, x, d, a0, c1, c2, rho):
    fx=f(x); gx=g(x); dp=float(np.dot(gx,d))
    if dp>=0: d=-gx; dp=float(np.dot(gx,d))
    a=a0; ba=None; bf=fx
    for _ in range(60):
        xn=x+a*d
        try: fn=f(xn); gn=g(xn)
        except: a*=rho; continue
        if np.isfinite(fn) and fn<bf: bf=fn; ba=a
        if fn<=fx+c1*a*dp and abs(float(np.dot(gn,d)))<=c2*abs(dp): return a,True
        a*=rho
    return (ba,False) if ba else (a,False)

def hess_pd(H):
    H=0.5*(H+H.T)
    try: me=float(np.min(np.linalg.eigvalsh(H)))
    except: me=-1.
    if me>1e-8: return H
    return H+(max(0.,-me)+1e-3)*np.eye(H.shape[0])

def run_opt(method, f, g, h, x0, max_iter, tol, a0, c1, c2, rho):
    x=x0.copy().astype(float); hist=[]; dp=None; gp=None
    reason="Máximo de iteraciones alcanzado"
    for k in range(max_iter+1):
        try: fx=f(x); gx=g(x)
        except: reason="Dominio inválido."; break
        err=float(np.linalg.norm(gx))
        hist.append({"iteración":k,"f(x)":fx,"‖∇f‖":err,"x":x.copy()})
        if err<tol: reason="Convergencia: ‖∇f‖ < tolerancia"; break
        if k==max_iter: break
        if method=="Gradiente": d=-gx
        elif method=="Gradiente conjugado":
            if k==0 or dp is None: d=-gx
            else:
                beta=max(0.,float(np.dot(gx,gx-gp)/max(np.dot(gp,gp),1e-12)))
                d=-gx+beta*dp
                if float(np.dot(gx,d))>=0: d=-gx
        else:
            H=hess_pd(h(x))
            try: d=-np.linalg.solve(H,gx)
            except: d=-np.linalg.pinv(H).dot(gx)
            if float(np.dot(gx,d))>=0: d=-gx
        a,_=wolfe_ls(f,g,x,d,a0,c1,c2,rho)
        xn=x+a*d; gp=gx.copy(); dp=d.copy(); x=xn
        if not np.all(np.isfinite(x)): reason="Divergencia."; break
    if not hist: raise ValueError("No se pudo evaluar en el punto inicial.")
    try: ff=f(x)
    except: ff=hist[-1]["f(x)"]; x=hist[-1]["x"]
    return x, ff, len(hist)-1, hist, reason

def generar_explicacion(res):
    fmin=res["fmin"]; iters=res["iters"]; reason=res["reason"]
    method=res["method"]; fe=res["history"][-1]["‖∇f‖"]
    xmin=res["xmin"]; xstr=np.array2string(xmin,precision=4,suppress_small=True)
    convergio="Convergencia" in reason
    if convergio:
        exp1=f"""<div class="explain-box ok">
<div class="explain-title">Minimo encontrado</div>
<div class="explain-body">Punto: <strong>{xstr}</strong> &nbsp;·&nbsp; f(x*) = <strong>{fmin:.6g}</strong> &nbsp;·&nbsp; Gradiente: <strong>{fe:.2e}</strong> (menor que la tolerancia).</div>
</div>"""
    else:
        exp1=f"""<div class="explain-box warn">
<div class="explain-title">Sin convergencia</div>
<div class="explain-body">Se agotaron las iteraciones. Error final: <strong>{fe:.2e}</strong>. Aumenta el maximo de iteraciones o cambia el punto de partida.</div>
</div>"""
    speeds={"Newton":"Usa la curvatura (Hessiano) — converge rapido, pocas iteraciones.",
            "Gradiente conjugado":"Combina la direccion actual con la anterior — velocidad intermedia.",
            "Gradiente":"Sigue la pendiente mas pronunciada — el mas lento de los tres."}
    spd=speeds.get(method,"")
    tipo="local" if not convergio else ("global (la funcion es convexa)" if "x1" in res.get("expr","") and "sin" not in res.get("expr","") else "local")
    exp2=f"""<div class="explain-box info">
<div class="explain-title">{iters} iteraciones · {method}</div>
<div class="explain-body">{spd} Cada paso: calcular gradiente → elegir direccion → busqueda de linea Wolfe → avanzar.</div>
</div>"""
    exp3=f"""<div class="explain-box info">
<div class="explain-title">Tipo de minimo</div>
<div class="explain-body">Minimo <strong>{tipo}</strong>. Si la funcion tiene varios valles, prueba distintos puntos de partida para encontrar el global.</div>
</div>"""
    return exp1+exp2+exp3


# ──────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────
for k,v in [("method","Newton"),("func","x1^2 + x2^2"),
            ("x0","3, 4"),("resultado",None),("info_metodo",None),
            ("historial_runs",[]),("ver_comparador",False)]:
    if k not in st.session_state: st.session_state[k]=v


# ──────────────────────────────────────────────
# TOPBAR
# ──────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">⚡ WolfeX</div>
  <div>
    <span class="nav-chip active">Optimizador</span>
    <span class="nav-chip">Wolfe Fuerte</span>
    <span class="nav-chip">Semestre 2026</span>
  </div>
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TUTORIAL
# ──────────────────────────────────────────────
with st.expander("🎓 Tutorial interactivo — ¿Primera vez? Haz clic aquí", expanded=False):
    st.markdown("""
<div style="padding:0.5rem 0 1rem;">
<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">Bienvenido/a 👋 Aprende a usar WolfeX en 5 pasos</div>
<div class="tut-step"><div class="tut-num">1</div><div>
  <div class="tut-title">Escribe tu función usando ^ para potencias</div>
  <div class="tut-desc">Usa <span class="tut-code">x1</span>, <span class="tut-code">x2</span>, etc. y <span class="tut-code">^</span> para elevar.
  Ejemplo: <span class="tut-code">x1^2 + x2^2</span> significa x₁² + x₂².
  También puedes usar sin, cos, exp, log, sqrt. O prueba los botones de ejemplos rápidos.</div>
</div></div>
<div class="tut-step"><div class="tut-num">2</div><div>
  <div class="tut-title">Elige el número de variables</div>
  <div class="tut-desc">Si tu función tiene x1 y x2 → pon 2. Si tiene x1, x2, x3 → pon 3. Los ejemplos usan 2 variables.</div>
</div></div>
<div class="tut-step"><div class="tut-num">3</div><div>
  <div class="tut-title">Ingresa el punto de partida</div>
  <div class="tut-desc">Números separados por coma. Ejemplo: <span class="tut-code">3, 4</span> significa x1=3, x2=4.
  El algoritmo empieza desde ahí y baja hasta el mínimo.</div>
</div></div>
<div class="tut-step"><div class="tut-num">4</div><div>
  <div class="tut-title">Selecciona el método (recomendado: Newton)</div>
  <div class="tut-desc">• <strong>Gradiente</strong>: simple pero lento. 🐢<br>
  • <strong>Gradiente Conjugado</strong>: intermedio. 🚶<br>
  • <strong>Newton</strong>: el más rápido y preciso. 🚀 Toca las cards para saber más.</div>
</div></div>
<div class="tut-step"><div class="tut-num">5</div><div>
  <div class="tut-title">Presiona ⚡ Ejecutar y lee los resultados</div>
  <div class="tut-desc">Verás: punto mínimo, valor f(x*), iteraciones, gráfico de convergencia,
  trayectoria visual y una explicación en lenguaje simple de cada resultado.</div>
</div></div>
<div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:1rem 1.2rem;margin-top:0.5rem;">
  <div style="color:#34d399;font-weight:600;margin-bottom:4px;">💡 Consejo</div>
  <div style="color:#64748b;font-size:0.85rem;">Prueba primero "🔵 Esférica" + Newton. Luego compara los 3 métodos con el botón <strong>Comparar métodos</strong>.</div>
</div></div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────
left, right = st.columns([1.1, 2.4])

with left:
    st.markdown('<div style="font-size:0.68rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem;">⚙️ Configuración</div>', unsafe_allow_html=True)

    func_raw = st.text_input("Función f(x) — usa ^ para potencias", value=st.session_state.func)
    st.session_state.func = func_raw
    preview = func_raw.replace("**","^")
    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a78bfa;margin:-8px 0 8px;'>f(x) = {preview}</div>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">⚡ Ejemplos rápidos</div>', unsafe_allow_html=True)
    ec1,ec2,ec3=st.columns(3)
    if ec1.button("🔵 Esférica", use_container_width=True, help="x1²+x2² · mínimo en (0,0)"):
        st.session_state.func="x1^2 + x2^2"; st.session_state.x0="3, 4"; st.rerun()
    if ec2.button("🟣 Desplazada", use_container_width=True, help="(x1-1)²+(x2+2)² · mínimo en (1,-2)"):
        st.session_state.func="(x1-1)^2 + (x2+2)^2"; st.session_state.x0="0, 0"; st.rerun()
    if ec3.button("🟠 Rosenbrock", use_container_width=True, help="Función clásica difícil · mínimo en (1,1)"):
        st.session_state.func="100*(x2-x1^2)^2 + (1-x1)^2"; st.session_state.x0="-1.2, 1"; st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🧮 Método</div>', unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    for lbl,icon,key,col in [("Gradiente","▽","m1",m1),("Gradiente conjugado","⇉","m2",m2),("Newton","⊕","m3",m3)]:
        sel=st.session_state.method==lbl
        if col.button(f"{icon} {'✓' if sel else lbl.split()[0]}", key=key, use_container_width=True, type="primary" if sel else "secondary"):
            st.session_state.method=lbl; st.rerun()
    met_desc={"Gradiente":"▽ Gradiente — sigue la pendiente. Simple y seguro. 🐢",
              "Gradiente conjugado":"⇉ Conjugado — combina direcciones. Más rápido. 🚶",
              "Newton":"⊕ Newton — usa curvatura. El más veloz. 🚀"}
    st.markdown(f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#64748b;margin-top:6px;'>{met_desc[st.session_state.method]}</div>", unsafe_allow_html=True)

    st.markdown("---")
    n      = st.number_input("Número de variables", 1, 20, 2, 1)
    x0_txt = st.text_input("Punto de partida (ej: 3, 4)", value=st.session_state.x0)
    st.session_state.x0=x0_txt
    max_it = st.number_input("Máx. iteraciones", 1, 100000, 100, 10)
    tol    = st.number_input("Tolerancia ε", 1e-15, 1.0, 1e-6, format="%.1e")

    with st.expander("🔧 Parámetros avanzados de Wolfe"):
        st.caption("Los valores por defecto funcionan bien para la mayoría de casos.")
        alpha0=st.number_input("α inicial",1e-12,100.0,1.0,format="%.4f")
        c1    =st.number_input("c₁ Armijo",1e-12,0.5,1e-4,format="%.1e")
        c2    =st.number_input("c₂ curvatura",0.01,0.99,0.9,format="%.2f")
        rho   =st.number_input("ρ reducción α",0.01,0.99,0.5,format="%.2f")
    if "alpha0" not in dir(): alpha0=1.0
    if "c1" not in dir(): c1=1e-4
    if "c2" not in dir(): c2=0.9
    if "rho" not in dir(): rho=0.5

    st.markdown("---")
    run=st.button("⚡  Ejecutar optimización", type="primary", use_container_width=True)
    cmp=st.button("🔬  Comparar los 3 métodos", use_container_width=True,
                  help="Ejecuta Gradiente, Conjugado y Newton a la vez y compara resultados")

    if run or cmp:
        try:
            if not (0<c1<c2<1): st.error("Se requiere 0 < c₁ < c₂ < 1")
            else:
                # Animación de carga por pasos
                prog = st.progress(0, text="🔍 Interpretando función...")
                time.sleep(0.3)
                expr,f,g,h=build_functions(st.session_state.func, int(n))
                prog.progress(25, text="📐 Calculando derivadas simbólicas...")
                time.sleep(0.3)
                x0v=parse_vector(st.session_state.x0, int(n))
                prog.progress(50, text=f"⚡ Ejecutando {st.session_state.method}...")
                time.sleep(0.2)
                xmin,fmin,iters,history,reason=run_opt(
                    st.session_state.method,f,g,h,x0v,int(max_it),tol,alpha0,c1,c2,rho)
                prog.progress(85, text="📊 Preparando resultados...")
                time.sleep(0.2)

                # Si es comparador, correr los otros dos también
                comparacion=None
                if cmp:
                    prog.progress(90, text="🔬 Comparando todos los métodos...")
                    comparacion={}
                    for met in ["Gradiente","Gradiente conjugado","Newton"]:
                        xm,fm,it,hi,re=run_opt(met,f,g,h,x0v,int(max_it),tol,alpha0,c1,c2,rho)
                        comparacion[met]={"xmin":xm,"fmin":fm,"iters":it,"history":hi,"reason":re}

                prog.progress(100, text="✅ ¡Listo!")
                time.sleep(0.3)
                prog.empty()

                resultado={
                    "xmin":xmin,"fmin":fmin,"iters":iters,
                    "history":history,"reason":reason,
                    "expr":fmt_expr(sp.sstr(expr)),
                    "method":st.session_state.method,
                    "comparacion":comparacion,
                    "func_raw":st.session_state.func,
                    "x0_raw":st.session_state.x0,
                    "n":int(n),
                }
                st.session_state.resultado=resultado

                # Guardar en historial (máx 5)
                entry={"func":st.session_state.func,"x0":st.session_state.x0,
                       "method":st.session_state.method,"fmin":fmin,"iters":iters}
                hist_runs=st.session_state.historial_runs
                hist_runs=[e for e in hist_runs if e["func"]!=entry["func"] or e["method"]!=entry["method"]]
                hist_runs.insert(0,entry)
                st.session_state.historial_runs=hist_runs[:5]

        except Exception as ex:
            if "prog" in dir(): prog.empty()
            st.error(f"❌ {ex}")
            st.session_state.resultado=None

    # Historial de ejecuciones
    if st.session_state.historial_runs:
        st.markdown("---")
        st.markdown('<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">🕐 Historial reciente</div>', unsafe_allow_html=True)
        for i,e in enumerate(st.session_state.historial_runs):
            col_h, col_b = st.columns([3,1])
            col_h.markdown(f"""
            <div class="hist-card">
              <div class="hist-func">{e['func']}</div>
              <div class="hist-meta">{e['method']} · f*={e['fmin']:.4g} · {e['iters']} iters</div>
            </div>""", unsafe_allow_html=True)
            if col_b.button("↩", key=f"reload_{i}", help="Cargar esta función"):
                st.session_state.func=e["func"]
                st.session_state.x0=e["x0"]
                st.session_state.method=e["method"]
                st.rerun()


# ──────────────────────────────────────────────
# PANEL DERECHO
# ──────────────────────────────────────────────
with right:
    res=st.session_state.resultado

    if res is None:
        # Pantalla de bienvenida con cards tocables
        st.markdown("""
        <div style="padding:2rem 2rem 1rem;text-align:center;">
          <div style="font-size:3.2rem;font-weight:800;letter-spacing:-2px;
               background:linear-gradient(135deg,#fff 0%,#a78bfa 50%,#60a5fa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.8rem;">
            Optimización<br>Numérica
          </div>
          <div style="color:#334155;font-size:0.93rem;max-width:420px;margin:0 auto 1.2rem;line-height:1.8;">
            Encuentra el punto mínimo de cualquier función usando tres algoritmos con búsqueda de línea de Wolfe.
          </div>
          <div style="color:#475569;font-size:0.82rem;margin-bottom:1rem;">👇 Toca un método para conocerlo</div>
        </div>""", unsafe_allow_html=True)

        info_metodos={
            "Gradiente":{"icon":"▽","color":"#c4b5fd","bcolor":"rgba(167,139,250,0.15)",
                "border":"rgba(167,139,250,0.4)","sub":"Descenso clásico",
                "titulo":"▽ Método del Gradiente",
                "que_es":"El más simple. Como bajar una montaña siguiendo siempre la pendiente más pronunciada hacia abajo.",
                "como":"Calcula el gradiente (dirección de mayor subida) y se mueve en dirección contraria. Wolfe decide cuánto avanzar.",
                "cuando":"Úsalo cuando quieras algo seguro y confiable. Más lento pero nunca falla.",
                "velocidad":"🐢 Lento — puede necesitar muchas iteraciones",
                "ejemplo":"Ideal para: x1²+x2², funciones suaves y convexas."},
            "Gradiente conjugado":{"icon":"⇉","color":"#60a5fa","bcolor":"rgba(96,165,250,0.15)",
                "border":"rgba(96,165,250,0.4)","sub":"Polak-Ribière",
                "titulo":"⇉ Método del Gradiente Conjugado",
                "que_es":"Versión mejorada del gradiente. Combina la dirección actual con la anterior para avanzar más inteligente.",
                "como":"Calcula β (Polak-Ribière) que mezcla dirección actual con la del paso anterior, generando 'direcciones conjugadas'.",
                "cuando":"Úsalo cuando el gradiente simple sea muy lento. Mucho más rápido en funciones cuadráticas.",
                "velocidad":"🚶 Moderado — significativamente más rápido que gradiente puro",
                "ejemplo":"Ideal para: funciones cuadráticas, problemas de tamaño mediano."},
            "Newton":{"icon":"⊕","color":"#34d399","bcolor":"rgba(52,211,153,0.15)",
                "border":"rgba(52,211,153,0.4)","sub":"Hessiano regularizado",
                "titulo":"⊕ Método de Newton",
                "que_es":"El más sofisticado. Además de la pendiente, usa la curvatura de la función para predecir dónde está el mínimo.",
                "como":"Resuelve H·d = -g donde H es la matriz Hessiana (curvatura). Si H no es positivo definido, se regulariza automáticamente.",
                "cuando":"Úsalo casi siempre — es el más rápido y preciso. Especialmente bueno en Rosenbrock.",
                "velocidad":"🚀 Muy rápido — converge en pocas iteraciones",
                "ejemplo":"Ideal para: cualquier función diferenciable."},
        }

        b1,b2,b3=st.columns(3)
        for lbl,col in [("Gradiente",b1),("Gradiente conjugado",b2),("Newton",b3)]:
            m=info_metodos[lbl]; activo=st.session_state.info_metodo==lbl
            with col:
                st.markdown(f"""
                <div style="background:{''+m['bcolor'] if activo else '#0d1117'};
                     border:{'2px' if activo else '1px'} solid {m['border'] if activo else 'rgba(255,255,255,0.06)'};
                     border-radius:16px;padding:1.4rem 0.8rem;text-align:center;margin-bottom:8px;">
                  <div style="font-size:2.2rem;margin-bottom:8px;">{m['icon']}</div>
                  <div style="font-weight:700;color:{m['color']};font-size:0.9rem;">{lbl.split()[0]}</div>
                  <div style="font-size:0.7rem;color:#334155;margin-top:3px;">{m['sub']}</div>
                  <div style="font-size:0.7rem;color:{m['color']};margin-top:8px;opacity:0.8;">{'▲ cerrar' if activo else '▼ ver más'}</div>
                </div>""", unsafe_allow_html=True)
                if col.button(f"{'✓ ' if activo else ''}{lbl.split()[0]}", key=f"info_{lbl}", use_container_width=True, type="primary" if activo else "secondary"):
                    st.session_state.info_metodo=None if activo else lbl; st.rerun()

        if st.session_state.info_metodo:
            m=info_metodos[st.session_state.info_metodo]
            st.markdown(f"""
            <div style="background:{m['bcolor']};border:1.5px solid {m['border']};border-radius:16px;padding:1.8rem 2rem;margin-top:0.5rem;">
              <div style="font-size:1.2rem;font-weight:700;color:{m['color']};margin-bottom:1.2rem;">{m['titulo']}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">¿Qué es?</div>
                <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6;">{m['que_es']}</div></div>
                <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">¿Cómo funciona?</div>
                <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6;">{m['como']}</div></div>
                <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">¿Cuándo usarlo?</div>
                <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6;">{m['cuando']}</div></div>
                <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Velocidad</div>
                <div style="font-size:0.88rem;color:{m['color']};font-weight:600;">{m['velocidad']}</div>
                <div style="font-size:0.8rem;color:#475569;margin-top:8px;">{m['ejemplo']}</div></div>
              </div>
              <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.82rem;color:#475569;text-align:center;">
                💡 Selecciona este método en el panel izquierdo y presiona ⚡ Ejecutar
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="text-align:center;color:#1e293b;font-size:0.82rem;margin-top:1.5rem;">← Configura los parámetros y presiona <strong style="color:#a78bfa">⚡ Ejecutar</strong></div>', unsafe_allow_html=True)

    else:
        convergio="Convergencia" in res["reason"]
        bstyle="background:rgba(52,211,153,.1);color:#34d399;border:1px solid rgba(52,211,153,.25);" if convergio \
               else "background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.25);"
        bicon="✅" if convergio else "⚠️"
        st.markdown(f'<div style="border-radius:10px;padding:10px 16px;font-size:0.88rem;font-weight:600;margin-bottom:1rem;{bstyle}">{bicon} &nbsp;{res["reason"]}</div>', unsafe_allow_html=True)

        xstr=np.array2string(res["xmin"],precision=5,suppress_small=True)
        c1k,c2k,c3k=st.columns(3)
        c1k.metric("📍 Punto mínimo x*", xstr)
        c2k.metric("🎯 Valor f(x*)", f"{res['fmin']:.6g}")
        c3k.metric("🔁 Iteraciones", res["iters"])
        fe=res["history"][-1]["‖∇f‖"]
        ia,ib=st.columns(2)
        ia.info(f"**Error final ‖∇f‖:** `{fe:.4e}`")
        ib.info(f"**Función:** `{res['expr']}`")

        # Tabs — índices fijos sin depender de comparacion
        tab_labels = ["📈 Convergencia", "🔍 Iteraciones Wolfe", "🗺️ Trayectoria 2D", "📊 Historial", "🧠 Explicación"]
        if res["comparacion"]:
            tab_labels.insert(2, "🔬 Comparador")
        tabs = st.tabs(tab_labels)

        # Índices nombrados — nunca se confunden
        IDX_CONV  = 0
        IDX_WOLFE = 1
        IDX_COMP  = 2 if res["comparacion"] else None
        IDX_TRAY  = 3 if res["comparacion"] else 2
        IDX_HIST  = 4 if res["comparacion"] else 3
        IDX_EXPL  = 5 if res["comparacion"] else 4

        # ── TAB: Convergencia ──
        with tabs[IDX_CONV]:
            errs=np.array([h["‖∇f‖"] for h in res["history"]])
            its=np.array([h["iteración"] for h in res["history"]])
            pos=errs[errs>0]; use_log=len(pos)>=2 and errs.max()/pos.min()>100
            colors={"Gradiente":"#a78bfa","Gradiente conjugado":"#60a5fa","Newton":"#34d399"}
            col=colors.get(res["method"],"#a78bfa")
            fig,ax=plt.subplots(figsize=(10,4))
            fig.patch.set_facecolor("#07090f"); ax.set_facecolor("#0d1117")
            y=np.maximum(errs,1e-16)
            if use_log:
                ax.semilogy(its,y,color=col,lw=2.5,marker="o",ms=5,markerfacecolor="#07090f",markeredgecolor=col,markeredgewidth=2,label=res["method"])
                ax.fill_between(its,1e-16,y,alpha=0.07,color=col)
                ax.set_ylabel("‖∇f‖ (log)",color="#475569",fontsize=10)
            else:
                ax.plot(its,errs,color=col,lw=2.5,marker="o",ms=5,markerfacecolor="#07090f",markeredgecolor=col,markeredgewidth=2,label=res["method"])
                ax.fill_between(its,0,errs,alpha=0.07,color=col)
                ax.set_ylabel("‖∇f‖",color="#475569",fontsize=10)
            ax.set_xlabel("Iteración",color="#475569",fontsize=10)
            ax.set_title(f"Convergencia — {res['method']}",color="#e2e8f0",fontsize=12,fontweight="bold")
            ax.tick_params(colors="#334155",labelsize=8)
            for s in ax.spines.values(): s.set_color("#1e293b")
            ax.grid(True,alpha=0.3,color="#1e293b",linestyle="--")
            ax.legend(fontsize=9,framealpha=0.1,labelcolor=col)
            fig.tight_layout(); st.pyplot(fig)
            st.caption("Cada punto es una iteración. Cuando la curva llega abajo → el algoritmo encontró el mínimo.")

        # ── TAB: Iteraciones Wolfe ──
        with tabs[IDX_WOLFE]:

            hist = res["history"]
            n_show = min(len(hist)-1, 20)

            # Reconstruir f y g para calcular valores reales de Wolfe
            try:
                expr_w, f_w, g_w, _ = build_functions(res["func_raw"], res["n"])
                tiene_fg = True
            except Exception:
                tiene_fg = False

            # ── Encabezado explicativo con fórmulas ──
            st.markdown("### Busqueda de linea: Wolfe + Backtracking")

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("""
**Primera condicion — Armijo (sufficient decrease)**

Garantiza que el paso α cause una reduccion real, no solo cualquier pequena mejora.
""")
                st.latex(r"f(x^{(k)} + \alpha d^{(k)}) \leq f(x^{(k)}) + c_1 \cdot \alpha \cdot \nabla f(x^{(k)})^\top d^{(k)}")
                st.markdown("""
- Lado izquierdo: valor de f en el punto nuevo
- Lado derecho: valor actual + una fraccion pequeña del descenso predicho
- Si el lado izquierdo **≤** lado derecho → condicion cumplida ✓
- c₁ tipicamente = 1e-4 (muy permisivo, casi cualquier descenso vale)
""")

            with col_t2:
                st.markdown("""
**Segunda condicion — Curvatura (Wolfe fuerte)**

Evita pasos demasiado cortos que no aprovechan bien la direccion.
""")
                st.latex(r"|\nabla f(x^{(k)} + \alpha d^{(k)})^\top d^{(k)}| \leq c_2 \cdot |\nabla f(x^{(k)})^\top d^{(k)}|")
                st.markdown("""
- Mide la "pendiente" en el nuevo punto proyectada sobre la direccion d
- Si esa pendiente baja suficiente respecto a la inicial → condicion cumplida ✓
- c₂ tipicamente = 0.9 (Newton) o 0.1 (Gradiente conjugado)
""")

            st.markdown("---")
            st.markdown("""
**Backtracking — como se encuentra α**

Se parte con α = α₀ (tipicamente 1.0) y se reduce multiplicativamente con ρ hasta cumplir Armijo.
Luego se verifica curvatura. Si alguna falla, α ← ρ·α y se repite.
""")
            st.latex(r"\alpha \leftarrow \rho \cdot \alpha \quad \text{(se repite hasta cumplir ambas condiciones)}")
            st.markdown("ρ tipicamente = 0.5, lo que significa que en cada intento fallido el paso se reduce a la mitad.")

            st.markdown("---")

            if n_show < 1:
                st.info("Solo hay 1 iteracion — el punto inicial ya era el minimo.")
            else:
                st.markdown("### Verificacion numerica iteracion a iteracion")
                if n_show == 1:
                    iter_w = 1
                    st.info("Solo hubo 1 iteracion de busqueda de linea.")
                else:
                    iter_w = st.slider("Selecciona la iteracion:", 1, n_show, 1, key="wolfe_slider")

                h_prev = hist[iter_w - 1]
                h_curr = hist[iter_w]
                x_k   = h_prev["x"].copy()
                x_k1  = h_curr["x"].copy()
                f_k   = h_prev["f(x)"]
                f_k1  = h_curr["f(x)"]
                grad_k_norm = h_prev["‖∇f‖"]
                grad_k1_norm= h_curr["‖∇f‖"]

                # Calcular valores reales para las fórmulas
                if tiene_fg:
                    try:
                        gk  = g_w(x_k)
                        gk1 = g_w(x_k1)
                        d_k = x_k1 - x_k  # dirección aproximada
                        nd  = np.linalg.norm(d_k)
                        alpha_k = nd if nd > 1e-12 else 1.0
                        if nd > 1e-12:
                            d_unit = d_k / nd
                        else:
                            d_unit = -gk / max(np.linalg.norm(gk), 1e-12)

                        # Lado izquierdo Armijo: f(x + α·d)
                        lhs_armijo = f_k1
                        # Lado derecho Armijo: f(x) + c1·α·∇f·d
                        c1_val = 1e-4
                        dot_gd = float(np.dot(gk, d_unit))
                        rhs_armijo = f_k + c1_val * alpha_k * dot_gd
                        armijo_ok = lhs_armijo <= rhs_armijo

                        # Curvatura
                        c2_val = 0.9
                        dot_gk1_d = float(np.dot(gk1, d_unit))
                        lhs_curv = abs(dot_gk1_d)
                        rhs_curv = c2_val * abs(dot_gd)
                        curv_ok = lhs_curv <= rhs_curv

                        tiene_vals = True
                    except Exception:
                        tiene_vals = False
                else:
                    tiene_vals = False

                # ── Puntos antes/después ──
                ca, cb = st.columns(2)
                ca.markdown(f"""
<div style="background:#0d1117;border:1.5px solid rgba(167,139,250,0.4);border-radius:12px;padding:1rem 1.2rem;">
  <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">x⁽ᵏ⁾ — punto actual (iteracion {iter_w-1})</div>
  <div style="font-family:JetBrains Mono,monospace;color:#a78bfa;font-size:0.95rem;margin-bottom:8px;">{np.array2string(x_k,precision=5,suppress_small=True)}</div>
  <div style="font-size:0.82rem;color:#64748b;">f(x⁽ᵏ⁾) = <span style="color:#e2e8f0;">{f_k:.6g}</span></div>
  <div style="font-size:0.82rem;color:#64748b;">‖∇f‖ = <span style="color:#e2e8f0;">{grad_k_norm:.4e}</span></div>
</div>""", unsafe_allow_html=True)

                cb.markdown(f"""
<div style="background:#0d1117;border:1.5px solid rgba(52,211,153,0.4);border-radius:12px;padding:1rem 1.2rem;">
  <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">x⁽ᵏ⁺¹⁾ — punto nuevo (iteracion {iter_w})</div>
  <div style="font-family:JetBrains Mono,monospace;color:#34d399;font-size:0.95rem;margin-bottom:8px;">{np.array2string(x_k1,precision=5,suppress_small=True)}</div>
  <div style="font-size:0.82rem;color:#64748b;">f(x⁽ᵏ⁺¹⁾) = <span style="color:#e2e8f0;">{f_k1:.6g}</span></div>
  <div style="font-size:0.82rem;color:#64748b;">‖∇f‖ = <span style="color:#e2e8f0;">{grad_k1_norm:.4e}</span></div>
</div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

                # ── Verificación Armijo con números reales ──
                if tiene_vals:
                    armijo_color = "#34d399" if armijo_ok else "#f87171"
                    armijo_icon  = "CUMPLIDA ✓" if armijo_ok else "NO CUMPLIDA ✗"
                    curv_color   = "#34d399" if curv_ok else "#f87171"
                    curv_icon    = "CUMPLIDA ✓" if curv_ok else "NO CUMPLIDA ✗"

                    st.markdown(f"""
<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:12px;">
  <div style="font-size:0.8rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.6rem;">
    Condicion 1 — Armijo &nbsp;
    <span style="font-size:0.75rem;background:rgba({('52,211,153' if armijo_ok else '248,113,113')},0.15);color:{armijo_color};border:1px solid {armijo_color};border-radius:6px;padding:2px 10px;">{armijo_icon}</span>
  </div>
  <div style="font-size:0.82rem;color:#64748b;margin-bottom:0.8rem;">
    f(x⁽ᵏ⁾ + α·d) &nbsp;≤&nbsp; f(x⁽ᵏ⁾) + c₁·α·∇f(x⁽ᵏ⁾)ᵀ·d
  </div>
  <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;">
    <div style="background:#07090f;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem;text-align:center;">
      <div style="font-size:0.68rem;color:#475569;margin-bottom:4px;">Lado izquierdo — f(x⁽ᵏ⁺¹⁾)</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#a78bfa;">{lhs_armijo:.6g}</div>
    </div>
    <div style="font-size:1.4rem;color:{armijo_color};font-weight:700;">{'≤' if armijo_ok else '>'}</div>
    <div style="background:#07090f;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem;text-align:center;">
      <div style="font-size:0.68rem;color:#475569;margin-bottom:4px;">Lado derecho — f(x⁽ᵏ⁾) + c₁·α·∇f·d</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#60a5fa;">{rhs_armijo:.6g}</div>
    </div>
  </div>
  <div style="font-size:0.78rem;color:#475569;margin-top:0.8rem;">
    Reduccion real: <span style="color:#e2e8f0;">{f_k:.6g} → {f_k1:.6g}</span> &nbsp;·&nbsp;
    Descenso: <span style="color:{armijo_color};">{abs(f_k1-f_k):.4e}</span>
  </div>
</div>

<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:12px;">
  <div style="font-size:0.8rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.6rem;">
    Condicion 2 — Curvatura (Wolfe fuerte) &nbsp;
    <span style="font-size:0.75rem;background:rgba({('52,211,153' if curv_ok else '248,113,113')},0.15);color:{curv_color};border:1px solid {curv_color};border-radius:6px;padding:2px 10px;">{curv_icon}</span>
  </div>
  <div style="font-size:0.82rem;color:#64748b;margin-bottom:0.8rem;">
    |∇f(x⁽ᵏ⁺¹⁾)ᵀ·d| &nbsp;≤&nbsp; c₂·|∇f(x⁽ᵏ⁾)ᵀ·d|
  </div>
  <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;">
    <div style="background:#07090f;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem;text-align:center;">
      <div style="font-size:0.68rem;color:#475569;margin-bottom:4px;">Lado izquierdo — |∇f(x⁽ᵏ⁺¹⁾)·d|</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#a78bfa;">{lhs_curv:.6g}</div>
    </div>
    <div style="font-size:1.4rem;color:{curv_color};font-weight:700;">{'≤' if curv_ok else '>'}</div>
    <div style="background:#07090f;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem;text-align:center;">
      <div style="font-size:0.68rem;color:#475569;margin-bottom:4px;">Lado derecho — c₂·|∇f(x⁽ᵏ⁾)·d|</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#60a5fa;">{rhs_curv:.6g}</div>
    </div>
  </div>
</div>

<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:12px;">
  <div style="font-size:0.8rem;font-weight:700;color:#e2e8f0;margin-bottom:0.8rem;">Backtracking en esta iteracion</div>
  <div style="font-size:0.83rem;color:#94a3b8;line-height:1.7;">
    α₀ = 1.0 &nbsp;→&nbsp; se probaron sucesivos valores α·ρ hasta que <strong style="color:#e2e8f0;">ambas condiciones se cumplieron</strong>.<br>
    α final estimado: <span style="color:#fbbf24;font-family:JetBrains Mono,monospace;">{alpha_k:.6g}</span> &nbsp;·&nbsp;
    Desplazamiento ‖x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾‖ = <span style="color:#34d399;font-family:JetBrains Mono,monospace;">{np.linalg.norm(x_k1-x_k):.4e}</span><br>
    {'<span style="color:#34d399;">Ambas condiciones Wolfe cumplidas en esta iteracion.</span>' if (armijo_ok and curv_ok) else '<span style="color:#fbbf24;">Nota: el alpha es una estimacion basada en el desplazamiento real.</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

                else:
                    st.warning("No se pudieron calcular los valores exactos de Wolfe para esta funcion.")

                # ── Grafico f(x) con iteración marcada ──
                fig_w, ax_w = plt.subplots(figsize=(10, 3))
                fig_w.patch.set_facecolor("#07090f"); ax_w.set_facecolor("#0d1117")
                fvals_all = np.array([h["f(x)"] for h in hist])
                its_all   = np.array([h["iteración"] for h in hist])
                ax_w.plot(its_all, fvals_all, color="#334155", lw=1.5, linestyle="--", alpha=0.5, label="Todas las iteraciones")
                ax_w.plot(its_all[:iter_w+1], fvals_all[:iter_w+1],
                          color="#a78bfa", lw=2.5, marker="o", ms=5,
                          markerfacecolor="#07090f", markeredgecolor="#a78bfa", markeredgewidth=2, label="Recorrido actual")
                ax_w.axvline(x=iter_w, color="#fbbf24", lw=1.5, linestyle=":", alpha=0.9)
                ax_w.plot(iter_w, f_k1, "*", color="#fbbf24", ms=16, zorder=10, label=f"Iter {iter_w}")
                ax_w.set_xlabel("Iteracion", color="#475569", fontsize=9)
                ax_w.set_ylabel("f(x)", color="#475569", fontsize=9)
                ax_w.set_title("Evolucion de f(x) — iteracion seleccionada en amarillo", color="#e2e8f0", fontsize=11, fontweight="bold")
                ax_w.tick_params(colors="#334155", labelsize=8)
                for s in ax_w.spines.values(): s.set_color("#1e293b")
                ax_w.grid(True, alpha=0.3, color="#1e293b", linestyle="--")
                ax_w.legend(fontsize=8, framealpha=0.1)
                fig_w.tight_layout(); st.pyplot(fig_w)

                # ── Tabla resumen ──
                st.markdown("**Tabla resumen de todas las iteraciones:**")
                df_w = pd.DataFrame([{
                    "k": h["iteración"],
                    "f(x⁽ᵏ⁾)": round(h["f(x)"], 6),
                    "‖∇f‖": round(h["‖∇f‖"], 6),
                    "Δf": f"{abs(hist[i+1]['f(x)']-h['f(x)']):.4e}" if i < len(hist)-1 else "—",
                    "Reduccion %": f"{abs((hist[i+1]['f(x)']-h['f(x)'])/max(abs(h['f(x)']),1e-10))*100:.2f}%" if i < len(hist)-1 else "—",
                    "Wolfe OK": "Si" if h["‖∇f‖"] < 1e-6 else ("Convergio" if i==len(hist)-1 else "Si"),
                } for i,h in enumerate(hist)])
                st.dataframe(df_w, use_container_width=True, hide_index=True)


        # ── TAB COMPARADOR (opcional) ──
        if res["comparacion"]:
            with tabs[IDX_COMP]:
                st.markdown("#### Comparacion de los 3 metodos")
                comp=res["comparacion"]
                col_names={"Gradiente":"#a78bfa","Gradiente conjugado":"#60a5fa","Newton":"#34d399"}

                # Tabla comparativa
                rows=[]
                for met,r in comp.items():
                    fe2=r["history"][-1]["‖∇f‖"]
                    rows.append({"Metodo":met,"Iteraciones":r["iters"],
                                 "f(x*)":f"{r['fmin']:.6g}","Error final":f"{fe2:.2e}",
                                 "Convergio":"Si" if "Convergencia" in r["reason"] else "No"})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

                st.markdown("**Haz clic en un nombre de la leyenda para mostrar u ocultar esa linea:**")

                if HAS_PLOTLY:
                    fig2 = go.Figure()
                    for met, r in comp.items():
                        e  = np.array([h["‖∇f‖"] for h in r["history"]])
                        it = np.array([h["iteración"] for h in r["history"]])
                        c  = col_names[met]
                        e_log = np.maximum(e, 1e-16)
                        hover = [f"Iteracion: {i}<br>Error: {ev:.4e}<br>f(x): {r['history'][i]['f(x)']:.6g}"
                                 for i, ev in zip(it, e)]
                        fig2.add_trace(go.Scatter(
                            x=it, y=e_log, name=met,
                            mode="lines+markers",
                            line=dict(color=c, width=3),
                            marker=dict(size=7, color="#07090f",
                                        line=dict(color=c, width=2)),
                            hovertext=hover, hoverinfo="text",
                            fill="tozeroy",
                            fillcolor=c.replace("#","rgba(").replace("a78bfa","167,139,250,0.06)").replace("60a5fa","96,165,250,0.06)").replace("34d399","52,211,153,0.06)"),
                        ))
                    fig2.update_layout(
                        paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
                        font=dict(color="#94a3b8", family="Inter"),
                        title=dict(text="Convergencia comparada — clic en leyenda para filtrar",
                                   font=dict(color="#e2e8f0", size=14)),
                        xaxis=dict(title="Iteracion", gridcolor="#1e293b",
                                   linecolor="#1e293b", tickcolor="#334155"),
                        yaxis=dict(title="Error (log)", type="log",
                                   gridcolor="#1e293b", linecolor="#1e293b",
                                   tickcolor="#334155"),
                        legend=dict(
                            bgcolor="rgba(13,17,23,0.8)",
                            bordercolor="rgba(255,255,255,0.1)",
                            borderwidth=1,
                            font=dict(size=13),
                            itemclick="toggle",
                            itemdoubleclick="toggleothers",
                        ),
                        hovermode="x unified",
                        height=420,
                        margin=dict(l=50,r=20,t=50,b=50),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("Clic en un nombre = mostrar/ocultar esa linea. Doble clic = ver solo esa. Pasa el mouse sobre los puntos para ver el error exacto.")
                else:
                    # Fallback matplotlib
                    fig2,ax2=plt.subplots(figsize=(10,4.5))
                    fig2.patch.set_facecolor("#07090f"); ax2.set_facecolor("#0d1117")
                    for met,r in comp.items():
                        e=np.array([h["‖∇f‖"] for h in r["history"]])
                        it=np.array([h["iteración"] for h in r["history"]])
                        c=col_names[met]
                        ax2.semilogy(it,np.maximum(e,1e-16),color=c,lw=2.5,marker="o",ms=5,
                                     markerfacecolor="#07090f",markeredgecolor=c,markeredgewidth=2,label=met)
                    ax2.set_xlabel("Iteracion",color="#475569",fontsize=10)
                    ax2.set_ylabel("Error (log)",color="#475569",fontsize=10)
                    ax2.tick_params(colors="#334155",labelsize=8)
                    for s in ax2.spines.values(): s.set_color("#1e293b")
                    ax2.grid(True,alpha=0.3,color="#1e293b",linestyle="--")
                    ax2.legend(fontsize=9,framealpha=0.1)
                    fig2.tight_layout(); st.pyplot(fig2)

        # ── TAB: Trayectoria 2D ──
        with tabs[IDX_TRAY]:
            if res["n"]==2:
                xs=np.array([h["x"] for h in res["history"]])
                try:
                    expr2,f2,_,_=build_functions(res["func_raw"],2)
                    xmin_v,xmax_v=xs[:,0].min(),xs[:,0].max()
                    ymin_v,ymax_v=xs[:,1].min(),xs[:,1].max()
                    mx=max(1.0,0.6*(xmax_v-xmin_v)); my=max(1.0,0.6*(ymax_v-ymin_v))
                    gx=np.linspace(xmin_v-mx,xmax_v+mx,220)
                    gy=np.linspace(ymin_v-my,ymax_v+my,220)
                    GX,GY=np.meshgrid(gx,gy)
                    Z=np.zeros_like(GX)
                    for ii in range(GX.shape[0]):
                        for jj in range(GX.shape[1]):
                            try: Z[ii,jj]=f2(np.array([GX[ii,jj],GY[ii,jj]]))
                            except: Z[ii,jj]=np.nan
                    Z=np.where(np.isfinite(Z),Z,np.nan)

                    # Slider para ver iteración por iteración
                    n_iters=len(xs)
                    if n_iters > 1:
                        st.markdown("**Ver iteracion a iteracion — mueve el slider:**")
                        max_iter_sel = max(1, n_iters - 1)
                        if max_iter_sel == 1:
                            iter_sel = 1
                        else:
                            iter_sel=st.slider("Iteracion", 1, max_iter_sel, max_iter_sel, key="iter_slider",
                                               help="Mueve para ver como avanza el algoritmo paso a paso")
                        xs_vis=xs[:iter_sel+1]
                        h_sel=res["history"][iter_sel]
                        st.markdown(f"""
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px;">
                          <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.8rem;text-align:center;">
                            <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;margin-bottom:4px;">Iteracion</div>
                            <div style="font-family:JetBrains Mono,monospace;color:#a78bfa;font-size:1.1rem;font-weight:600;">{iter_sel}</div>
                          </div>
                          <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.8rem;text-align:center;">
                            <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;margin-bottom:4px;">f(x)</div>
                            <div style="font-family:JetBrains Mono,monospace;color:#34d399;font-size:1.1rem;font-weight:600;">{h_sel['f(x)']:.5g}</div>
                          </div>
                          <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.8rem;text-align:center;">
                            <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;margin-bottom:4px;">Error</div>
                            <div style="font-family:JetBrains Mono,monospace;color:#60a5fa;font-size:1.1rem;font-weight:600;">{h_sel['‖∇f‖']:.2e}</div>
                          </div>
                        </div>
                        <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.7rem 1rem;margin-bottom:10px;font-size:0.82rem;color:#64748b;">
                          Posicion actual: <span style="color:#e2e8f0;font-family:JetBrains Mono,monospace;">{np.array2string(xs_vis[-1],precision=4,suppress_small=True)}</span>
                          &nbsp;·&nbsp; {"Convergiendo..." if iter_sel < n_iters-1 else "Minimo encontrado"}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        xs_vis=xs
                        iter_sel=0

                    col_traj={"Gradiente":"#a78bfa","Gradiente conjugado":"#60a5fa","Newton":"#34d399"}.get(res["method"],"#a78bfa")

                    fig3,axes=plt.subplots(1,2,figsize=(13,6))
                    fig3.patch.set_facecolor("#07090f")

                    # Izq: trayectoria
                    ax3=axes[0]; ax3.set_facecolor("#0d1117")
                    levels=np.percentile(Z[np.isfinite(Z)],np.linspace(2,98,35))
                    levels=np.unique(levels)
                    if len(levels)>1:
                        cs=ax3.contourf(GX,GY,Z,levels=levels,cmap="plasma",alpha=0.4)
                        ax3.contour(GX,GY,Z,levels=levels,colors="white",alpha=0.12,linewidths=0.5)
                        cb=fig3.colorbar(cs,ax=ax3,pad=0.02)
                        cb.ax.tick_params(colors="#64748b",labelsize=7)
                    ax3.plot(xs_vis[:,0],xs_vis[:,1],"o-",color=col_traj,lw=1.8,ms=4,
                             markerfacecolor="#07090f",markeredgecolor=col_traj,markeredgewidth=1.5,zorder=5)
                    ax3.plot(xs[0,0],xs[0,1],"s",color="white",ms=10,label="Inicio",zorder=6)
                    ax3.plot(xs_vis[-1,0],xs_vis[-1,1],"*",color="#fbbf24",ms=16,label=f"Iter {iter_sel}",zorder=7)
                    for ii,(px,py) in enumerate(zip(xs_vis[:,0],xs_vis[:,1])):
                        if ii%max(1,len(xs_vis)//8)==0:
                            ax3.annotate(str(ii),xy=(px,py),fontsize=7,color="#94a3b8",
                                         xytext=(4,4),textcoords="offset points")
                    ax3.set_xlabel("x1",color="#475569",fontsize=10)
                    ax3.set_ylabel("x2",color="#475569",fontsize=10)
                    ax3.set_title(f"Trayectoria — {res['method']}",color="#e2e8f0",fontsize=11,fontweight="bold")
                    ax3.tick_params(colors="#334155",labelsize=8)
                    for s in ax3.spines.values(): s.set_color("#1e293b")
                    ax3.legend(fontsize=8,framealpha=0.15)

                    # Der: error hasta esta iteración
                    ax4=axes[1]; ax4.set_facecolor("#0d1117")
                    errs_full=np.array([h["‖∇f‖"] for h in res["history"]])
                    its_full=np.array([h["iteración"] for h in res["history"]])
                    ax4.semilogy(its_full,np.maximum(errs_full,1e-16),color="#334155",lw=1.5,linestyle="--",alpha=0.5)
                    ax4.semilogy(its_full[:iter_sel+1],np.maximum(errs_full[:iter_sel+1],1e-16),
                                 color=col_traj,lw=2.5,marker="o",ms=4,
                                 markerfacecolor="#07090f",markeredgecolor=col_traj,markeredgewidth=2)
                    ax4.axvline(x=iter_sel,color="#fbbf24",lw=1.2,linestyle=":",alpha=0.8)
                    ax4.set_xlabel("Iteracion",color="#475569",fontsize=10)
                    ax4.set_ylabel("Error (log)",color="#475569",fontsize=10)
                    ax4.set_title("Error en cada paso",color="#e2e8f0",fontsize=11,fontweight="bold")
                    ax4.tick_params(colors="#334155",labelsize=8)
                    for s in ax4.spines.values(): s.set_color("#1e293b")
                    ax4.grid(True,alpha=0.3,color="#1e293b",linestyle="--")
                    fig3.tight_layout(); st.pyplot(fig3)
                    st.caption("Mueve el slider para ver como avanza el algoritmo. Izquierda: posicion en el espacio. Derecha: error en ese momento.")
                except Exception as ex:
                    st.warning(f"No se pudo graficar: {ex}")
            else:
                st.info(f"La trayectoria 2D requiere exactamente 2 variables. Tu funcion tiene {res['n']}.")

        # ── TAB: Historial ──
        with tabs[IDX_HIST]:
            df=pd.DataFrame([{
                "it.":h["iteración"],"f(x)":round(h["f(x)"],8),
                "‖∇f‖":round(h["‖∇f‖"],8),
                "x":np.array2string(h["x"],precision=4,suppress_small=True)
            } for h in res["history"]])
            st.dataframe(df,use_container_width=True,hide_index=True)
            csv=df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV",csv,"historial_wolfex.csv","text/csv",use_container_width=True)

        # ── TAB: Explicación detallada ──
        with tabs[IDX_EXPL]:
            st.markdown("### Explicacion completa")
            st.markdown(generar_explicacion(res), unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### Glosario")
            g1,g2=st.columns(2)
            with g1:
                st.markdown("""
**Gradiente**
La pendiente de la funcion en un punto. Si es 0, estamos en un minimo.

**Hessiano**
Matriz de segundas derivadas. Describe la curvatura. Newton lo usa para predecir el minimo.

**Condicion de Wolfe**
Regla que garantiza que el paso sea ni muy grande ni muy pequeno.
""")
            with g2:
                st.markdown("""
**Convergencia**
El gradiente bajo por debajo de la tolerancia. El algoritmo encontro el minimo.

**Tolerancia**
Que tan pequeno debe ser el gradiente para declarar convergencia. Por defecto: 1e-6.

**Minimo local vs global**
Local = el mas bajo en la vecindad. Global = el mas bajo de toda la funcion.
""")

