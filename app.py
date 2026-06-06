import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Optimizador WolfeX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Diseño visual
# ------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    section[data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero-card {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero-card h1 {
        color: white;
        font-size: 2.4rem;
        margin-bottom: 0.35rem;
    }
    .hero-card p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .info-card {
        padding: 1.1rem 1.25rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .small-title {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .example-code {
        background: #eff6ff;
        color: #1e40af;
        padding: 0.18rem 0.45rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Utilidades
# ------------------------------
def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip() != ""]
        if len(vals) != n:
            raise ValueError(f"Debe ingresar exactamente {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto de partida inválido: {e}")


def normalize_expression(expr_text):
    """Permite escribir potencias con ^ en vez de **."""
    return expr_text.replace("^", "**").replace("π", "pi")


def build_functions(expr_text, n):
    x_symbols = sp.symbols("x1:" + str(n + 1))
    local_dict = {f"x{i+1}": x_symbols[i] for i in range(n)}
    local_dict.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
        "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
        "abs": sp.Abs
    })
    try:
        expr = sp.sympify(normalize_expression(expr_text), locals=local_dict)
        grad_expr = [sp.diff(expr, v) for v in x_symbols]
        hess_expr = sp.hessian(expr, x_symbols)
        f_np = sp.lambdify((x_symbols,), expr, "numpy")
        g_np = sp.lambdify((x_symbols,), grad_expr, "numpy")
        h_np = sp.lambdify((x_symbols,), hess_expr, "numpy")
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función objetivo: {e}")

    def f(x):
        return float(np.asarray(f_np(tuple(x)), dtype=float))

    def g(x):
        return np.array(g_np(tuple(x)), dtype=float).reshape(-1)

    def h(x):
        return np.array(h_np(tuple(x)), dtype=float)

    return expr, f, g, h


def wolfe_line_search(f, g, x, d, alpha0=1.0, c1=1e-4, c2=0.9, rho=0.5, max_ls=50):
    alpha = alpha0
    fx = f(x)
    gx = g(x)
    phi0_der = float(np.dot(gx, d))

    if phi0_der >= 0:
        d = -gx
        phi0_der = float(np.dot(gx, d))

    for _ in range(max_ls):
        x_new = x + alpha * d
        try:
            f_new = f(x_new)
            g_new = g(x_new)
        except Exception:
            alpha *= rho
            continue

        cond1 = f_new <= fx + c1 * alpha * phi0_der
        cond2 = float(np.dot(g_new, d)) >= c2 * phi0_der

        if cond1 and cond2:
            return alpha, True
        alpha *= rho

    return alpha, False


def optimize(method, f, g, h, x0, max_iter, tol, alpha0, c1, c2, rho):
    x = x0.astype(float).copy()
    history = []
    d_prev = None
    g_prev = None
    stop_reason = "Máximo de iteraciones alcanzado"

    for k in range(max_iter + 1):
        fx = f(x)
        gx = g(x)
        error = float(np.linalg.norm(gx))
        history.append({"iteración": k, "f(x)": fx, "error ||grad||": error, "x": x.copy()})

        if error < tol:
            stop_reason = "Convergencia alcanzada: ||gradiente|| < tolerancia"
            break

        if k == max_iter:
            break

        if method == "Gradiente":
            d = -gx

        elif method == "Gradiente conjugado":
            if k == 0 or d_prev is None or g_prev is None:
                d = -gx
            else:
                beta = max(0.0, float(np.dot(gx, gx - g_prev) / max(np.dot(g_prev, g_prev), 1e-12)))
                d = -gx + beta * d_prev
                if float(np.dot(gx, d)) >= 0:
                    d = -gx

        elif method == "Newton":
            H = h(x)
            try:
                d = -np.linalg.solve(H, gx)
            except np.linalg.LinAlgError:
                d = -np.linalg.pinv(H).dot(gx)
            if float(np.dot(gx, d)) >= 0:
                d = -gx
        else:
            raise ValueError("Método no reconocido")

        alpha, wolfe_ok = wolfe_line_search(f, g, x, d, alpha0, c1, c2, rho)
        x_new = x + alpha * d

        g_prev = gx.copy()
        d_prev = d.copy()
        x = x_new

        if not np.all(np.isfinite(x)):
            stop_reason = "El método generó valores no finitos"
            break

    return x, f(x), len(history) - 1, history, stop_reason


def format_point(x):
    values = ", ".join([f"x{i+1} = {val:.6g}" for i, val in enumerate(x)])
    return f"({values})"


# ------------------------------
# Interfaz
# ------------------------------
with st.sidebar:
    st.markdown("### Datos de entrada")
    n = st.number_input("Número de variables", min_value=1, max_value=5, value=2, step=1)
    method = st.selectbox("Método de optimización", ["Gradiente", "Gradiente conjugado", "Newton"])
    function_text = st.text_input("Función objetivo", value="x1^2 + x2^2", help="Use ^ para potencias. Ejemplo: x1^2 + x2^2")
    x0_text = st.text_input("Punto de partida", value="3, 4", help="Separe los valores con coma. Ejemplo: 3, 4")
    max_iter = st.number_input("Número máximo de iteraciones", min_value=1, max_value=10000, value=100, step=10)
    tol = st.number_input("Tolerancia de convergencia", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e")

    st.markdown("### Parámetros Wolfe")
    alpha0 = st.number_input("Alpha inicial", min_value=1e-12, max_value=100.0, value=1.0, format="%.4f")
    c1 = st.number_input("c1 primera condición", min_value=1e-12, max_value=0.5, value=1e-4, format="%.1e")
    c2 = st.number_input("c2 segunda condición", min_value=0.01, max_value=0.99, value=0.9, format="%.2f")
    rho = st.number_input("Factor de reducción alpha", min_value=0.01, max_value=0.99, value=0.5, format="%.2f")
    run = st.button("Ejecutar optimización", type="primary")

st.markdown("""
<div class="hero-card">
    <h1>Optimizador Multivariable con Condiciones de Wolfe</h1>
    <p>Proyecto Final de Métodos de Optimización · Gradiente · Gradiente Conjugado · Newton</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    <div class="info-card">
        <div class="small-title">Métodos incluidos</div>
        Gradiente, Gradiente Conjugado y Newton.
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="info-card">
        <div class="small-title">Búsqueda de línea</div>
        Primera y segunda condición de Wolfe.
    </div>
    """, unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="info-card">
        <div class="small-title">Valor agregado</div>
        Tabla de iteraciones, gráfico y descarga CSV.
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Instrucciones de uso")
st.write("Ingrese la función objetivo usando variables `x1`, `x2`, `x3`, etc. Para potencias use el símbolo `^`.")

st.markdown("""
Ejemplos de función:
- <span class="example-code">x1^2 + x2^2</span>
- <span class="example-code">(x1-1)^2 + (x2+2)^2</span>
- <span class="example-code">100*(x2-x1^2)^2 + (1-x1)^2</span>
""", unsafe_allow_html=True)

if run:
    try:
        if not (0 < c1 < c2 < 1):
            st.error("Debe cumplirse 0 < c1 < c2 < 1 para las condiciones de Wolfe.")
            st.stop()

        expr, f, g, h = build_functions(function_text, int(n))
        x0 = parse_vector(x0_text, int(n))
        xmin, fmin, iters, history, stop_reason = optimize(method, f, g, h, x0, int(max_iter), tol, alpha0, c1, c2, rho)

        st.success("Optimización finalizada correctamente")
        col1, col2, col3 = st.columns(3)
        col1.metric("Punto mínimo encontrado", format_point(xmin))
        col2.metric("Valor mínimo f(x*)", f"{fmin:.10g}")
        col3.metric("Iteraciones realizadas", iters)

        final_error = history[-1]["error ||grad||"]
        st.markdown("### Resumen del criterio de parada")
        st.write(f"**Error final:** `{final_error:.10g}`")
        st.write(f"**Criterio de parada:** {stop_reason}")
        st.write(f"**Función interpretada:** `{sp.sstr(expr).replace('**', '^')}`")

        df = pd.DataFrame([
            {
                "iteración": item["iteración"],
                "f(x)": item["f(x)"],
                "error ||grad||": item["error ||grad||"],
                "x": np.array2string(item["x"], precision=6)
            }
            for item in history
        ])

        st.markdown("### Tabla de iteraciones")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Gráfico de convergencia")
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.plot(df["iteración"], df["error ||grad||"], marker="o", linewidth=2)
        ax.set_xlabel("Número de iteración")
        ax.set_ylabel("Error ||gradiente||")
        ax.set_title("Error versus número de iteraciones")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar historial de iteraciones en CSV", data=csv, file_name="historial_optimizacion.csv", mime="text/csv")

        st.info("Valor agregado: la aplicación permite descargar el historial de iteraciones para revisar el proceso paso a paso.")

    except Exception as e:
        st.error(str(e))
else:
    st.info("Complete los parámetros en la barra lateral y presione Ejecutar optimización.")
