# TFG_ETH_FORECAST

**Sistema híbrido de apoyo a la toma de decisiones en el mercado de Ethereum, basado en la orquestación de un modelo de regímenes (HMM), una red neuronal recurrente (LSTM) y un sistema de recuperación aumentada por generación (RAG).**

Trabajo de Fin de Grado — Ingeniería Matemática.

---

## 1. Introducción

Este proyecto nació con la idea inicial de "predecir el precio de Ethereum". Sin embargo, tras una experimentación rigurosa, el trabajo evolucionó hacia algo más sólido y honesto: **demostrar que la dirección del precio a corto plazo no es predecible de forma fiable a partir de datos históricos, y construir en su lugar un sistema de apoyo a la decisión** que combina señales cuantitativas con interpretación cualitativa.

La aportación principal del trabajo **no es la predicción en sí**, sino la **arquitectura híbrida** que orquesta tres componentes complementarios:

- **HMM** → identifica el **régimen de mercado** (alcista, bajista o acumulación).
- **LSTM** → estima la **evolución esperada del precio** a corto plazo (magnitud orientativa; dirección poco fiable).
- **RAG** → incorpora **contexto histórico, noticias y conocimiento experto** para interpretar las señales anteriores y generar una conclusión razonada.

Dicho de otro modo: el HMM y la LSTM generan **señales cuantitativas**, mientras que el RAG aporta **contexto cualitativo**. Ninguno de los tres por separado resuelve el problema; el valor está en su combinación.

---

## 2. Objetivos

1. Diseñar y entrenar un **modelo de regímenes de mercado (HMM)** que clasifique el estado de Ethereum en fases interpretables.
2. Construir y evaluar de forma rigurosa una **red LSTM** para estimar el retorno a corto plazo, documentando honestamente sus límites.
3. Integrar un **sistema RAG** que recupere e interprete información contextual (noticias, conocimiento experto) para apoyar la toma de decisiones.
4. **Orquestar los tres bloques** en un sistema coherente y automatizado.
5. Dejar el proyecto **reproducible y automatizado** (actualización diaria de datos vía GitHub Actions) y preparar una aplicación sencilla para la defensa.

---

## 3. Arquitectura del sistema

```
                  ┌─────────────────────────────────────────────┐
                  │          DATOS DE MERCADO (diarios)          │
                  │  CoinGecko (BTC/ETH, dominancias, mcap)      │
                  │  Alternative.me (Fear & Greed)               │
                  └─────────────────────┬───────────────────────┘
                                        │
                          df_merged.csv (serie histórica)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
        ┌───────────┐            ┌─────────────┐          ┌──────────────┐
        │   HMM     │            │  FEATURES   │          │   (macro)    │
        │ regímenes │──────────▶ │  df_model   │          │  yfinance    │
        └───────────┘  regimenes └──────┬──────┘          │ (en el RAG)  │
                          .csv          │                 └──────┬───────┘
                                        ▼                        │
                                  ┌───────────┐                  │
                                  │   LSTM    │                  │
                                  │  señal 3d │                  │
                                  └─────┬─────┘                  │
                                        │                        │
                                        ▼                        ▼
                          ┌──────────────────────────────────────────┐
                          │                  RAG                      │
                          │  señal LSTM + régimen HMM + snapshot +    │
                          │  noticias (API) + TXT (contexto/experto)  │
                          │            ↓                              │
                          │     LLM (Gemini / Claude) → respuesta     │
                          └──────────────────────────────────────────┘
```

---

## 4. Componente 1 — HMM (modelo de regímenes)

**Estado: COMPLETADO y CERRADO.**

### Qué hace
Clasifica cada día del mercado de Ethereum en uno de **tres regímenes**:
- **Acumulación** (lateral / consolidación)
- **Bajista**
- **Alcista**

### Variables de entrada (5)
El modelo se entrenó sobre **5 variables escaladas**, elegidas por su sentido económico:

| Variable | Descripción |
|----------|-------------|
| `vol_30d` | Volatilidad a 30 días |
| `cum_ret_60d` | Retorno acumulado a 60 días |
| `dist_sma200` | Distancia a la media móvil de 200 sesiones |
| `drawdown` | Caída desde el último máximo |
| `fg` | Índice Fear & Greed (cruda, 0–100) |

### Detalle técnico importante
- La cadena es: **5 variables → StandardScaler → GaussianHMM (3 estados, covarianza full)**.
- **El PCA NO se usa para clasificar.** El `.pkl` contiene un objeto PCA, pero es **solo para visualización 2D** de los estados. El HMM se entrena y predice directamente sobre las 5 variables escaladas (`hmm.n_features == 5`).
- Se usan ventanas largas (60d, sma200) para que los regímenes sean **estables** y no "parpadeen" dentro de una misma tendencia.

### Validación
Los regímenes detectados se cruzaron con **episodios históricos reales** (cripto-invierno 2018, verano DeFi 2020, bear 2022 con LUNA/FTX, recuperación 2023, máximos 2025) y son coherentes. Reparto global: **Acumulación ~39%, Bajista ~36%, Alcista ~25%**, con ~28 tramos.

### Archivo del modelo
`models/hmm_final.pkl` → dict con `{hmm, scaler, pca, vars_regimen, nombres}`.
Mapeo de estados: `{0: 'Acumulacion', 1: 'Bajista', 2: 'Alcista'}` (asignado por características, no por número fijo).

---

## 5. Componente 2 — LSTM (señal cuantitativa)

**Estado: COMPLETADO y CERRADO.**

### Qué hace
Estima el **retorno de Ethereum** (`eth_close_ret`) para los **próximos 3 días** (HORIZON=3), a partir de una ventana de los **30 días previos** (SEQ_LEN=30).

### Proceso de experimentación
Se realizó una **búsqueda masiva** (varios miles de combinaciones de variables y arquitecturas, desde redes mínimas de una capa hasta redes mayores de varias capas). Conclusión clave:

> **Todas las configuraciones convergen al mismo techo de error (val_loss ≈ 0,256–0,266).** El acierto direccional se mantiene en torno al **50–52 %**, indistinguible del azar.

### Conjunto de variables final (16 + régimen)
Elegido **cruzando los dos análisis** (las que mejor predicen la magnitud + las que mejor captan la dirección), por frecuencia de aparición en los mejores modelos (no por mínima pérdida puntual, para evitar sobreajuste de selección):

`eth_cum_ret_30d`, `btc_dominance_chg14d`, `inflation_chg30`, `eth_bb_width`, `eth_mfi14`, `eth_dist_sma200`, `alt_dominance_diff`, `eth_rsi14`, `eth_vol_14d`, `eth_mcap_ret`, `eth_stoch_d`, `n_miedo_ext_30d`, `n_codicia_15d`, `presion_ext_neta_15d`, `fear_greed_scaled`, `eth_close_ret` **+ régimen one-hot (regime_0/1/2)** = **19 entradas**.

### Arquitectura final
- **Una sola capa LSTM de 32 neuronas** (`hidden_sizes=(32,)`), dropout 0.35.
- Se verificó que una red de 1 capa **iguala** a una de 2 capas (parsimonia: el límite no es de capacidad del modelo).
- Entrenamiento: Huber loss + Adam + ReduceLROnPlateau + early stopping.
- Escalado selectivo: variables continuas con **RobustScaler**, target con **StandardScaler**, régimen one-hot **sin escalar**.
- Split temporal **70/15/15** (sin fuga de datos).

### Resultado honesto en test (evaluación única)
| Métrica | Valor | Baseline (predecir 0) |
|---------|-------|----------------------|
| val_loss | 0,2647 (mejor en época 0) | — |
| MAE | 2,52 % | 2,52 % |
| RMSE | 3,67 % | 3,67 % |
| DirAcc | **0,51** | 0,50 (azar) |

**Interpretación:** la convergencia inmediata del val_loss y el acierto direccional cercano al azar **sugieren que la información puramente histórica es insuficiente para obtener una ventaja predictiva con una LSTM convencional** (consistente con la hipótesis del mercado eficiente). Esto **no es un fracaso, es el hallazgo**: justifica que el sistema no se apoye solo en la red, sino que la complemente con el RAG.

### Diferencia entrenamiento / producción
- En el **entrenamiento** de la LSTM se usó la versión **causal** del régimen (para no filtrar futuro).
- En **producción**, el régimen se calcula con el HMM completo (clasificar el día actual con el modelo entrenado es legítimo).

### Archivos del modelo
- `models/lstm_final.pt` → pesos + arquitectura + lista de columnas.
- `models/scaler_lstm.pkl` → scalers (sx, sy) + columnas.

### Concepto clave de la señal
Los 3 retornos son **diarios y encadenados** (cada día respecto al anterior, no respecto a hoy). Los precios estimados se obtienen **multiplicando**: `precio × (1+r1) × (1+r2) × (1+r3)`, no sumando.

---

## 6. Componente 3 — RAG (contexto cualitativo)

**Estado: EN CONSTRUCCIÓN dentro de este repo.**

### Qué hace (y qué NO hace)
El RAG **no es un modelo predictivo**. Es un **sistema de apoyo a la decisión** que:
1. Recupera información textual relevante (noticias frescas vía API, conocimiento experto en TXT).
2. Recibe la **señal cuantitativa** (predicción LSTM + régimen HMM + snapshot del mercado).
3. Razona una **conclusión en lenguaje natural**, interpretando las señales con el contexto.

Las noticias se obtienen mediante **llamadas a APIs dentro del propio notebook** (≈200 noticias); no hay un CSV de noticias en el repo.

### Documentos de contexto (TXT)
Ubicados en `data/txt_rag/`, en dos categorías:
- **`fijos/`** — conocimiento estable que se inyecta siempre en el prompt:
  - `contexto_ethereum.txt`
  - `contexto_mercado_eth.txt`
  - `mentalidad_buffet.txt` (mentalidad *value investing* para contrarrestar el sesgo optimista)
  - `motor_razonamiento_mercado.txt`
- **`embebidos/`** — documentos largos para trocear e indexar vectorialmente (de momento solo `historia_ethereum.txt`; el sistema de ChromaDB aún **no está montado**).

### Modelo de generación
- **Embeddings:** Google (`gemini-embedding-001`).
- **Generación:** actualmente **Gemini**; la función de generación está **aislada** para poder enchufar **Claude (API Anthropic, modelo Opus)** cambiando un solo parámetro.

### Sesgo reconocido (limitación)
El RAG tiene cierto **sesgo optimista** (los textos parten de una visión que cree en el valor de las criptomonedas). Se contrarresta con la mentalidad *value/Buffett*, pero se reconoce explícitamente como limitación.

---

## 7. Pipeline de datos (automatizado)

**Estado: FUNCIONANDO en GitHub Actions.**

Cada día (cron 09:00 UTC) se ejecutan en orden tres scripts ligeros:

1. **`src/data/actualizar_diario.py`** — descarga datos de mercado (CoinGecko + Alternative.me), hace backup de `df_merged` y lo actualiza hasta hoy.
2. **`src/data/actualizar_regimenes.py`** — carga `hmm_final.pkl`, calcula las 5 variables, escala y clasifica el régimen. Regenera `regimenes.csv`.
3. **`src/data/calcular_features.py`** — junta `df_merged` + `regimenes`, calcula todas las features (~96) + régimen one-hot = ~99 columnas. Genera `df_model.csv`.

La **predicción de la LSTM** (`predecir_lstm.py`) **no** va en el cron (requiere PyTorch, pesado): se ejecuta **bajo demanda** cuando el usuario pregunta al sistema.

> **Reproducibilidad:** las versiones de `requirements-datos.txt` están **fijadas** (scikit-learn 1.8.0, numpy 2.4.0, scipy 1.16.3, hmmlearn 0.3.3, joblib 1.5.3) para que el `.pkl` del HMM se cargue idéntico al entorno de entrenamiento. Sin fijar versiones, el modelo daba error de carga.

---

## 8. Estructura del repositorio

```
TFG_ETH_FORECAST/
│
├── .github/
│   └── workflows/
│       └── actualizar_diario.yaml      # workflow diario (3 scripts en orden + commit/push)
│
├── data/
│   ├── csv/
│   │   ├── raw/
│   │   │   ├── df_merged.csv            # serie histórica de mercado (fuente base)
│   │   │   ├── regimenes.csv            # régimen por día (salida del HMM)
│   │   │   └── resultados_modelos.csv   # log de la búsqueda masiva de la LSTM
│   │   ├── processed/
│   │   │   └── df_model.csv             # features completas (~99 col) → lo consume la LSTM
│   │   └── backups/
│   │       └── df_merged_backup_*.csv   # copias de seguridad de df_merged
│   │
│   └── txt_rag/                         # documentos de conocimiento para el RAG
│       ├── fijos/                       # contexto estable (siempre en el prompt)
│       │   ├── contexto_ethereum.txt
│       │   ├── contexto_mercado_eth.txt
│       │   ├── mentalidad_buffet.txt
│       │   └── motor_razonamiento_mercado.txt
│       └── embebidos/                   # documentos largos para indexar (ChromaDB pendiente)
│           └── historia_ethereum.txt
│
├── models/
│   ├── hmm_final.pkl                    # HMM (hmm + scaler + pca[viz] + nombres)
│   ├── lstm_final.pt                    # LSTM (pesos + arquitectura + columnas)
│   └── scaler_lstm.pkl                  # scalers de la LSTM (sx, sy)
│
├── src/
│   └── data/
│       ├── actualizar_diario.py         # 1. descarga y actualiza df_merged
│       ├── actualizar_regimenes.py      # 2. calcula regímenes (HMM)
│       ├── calcular_features.py         # 3. genera df_model
│       └── predecir_lstm.py             # señal LSTM (bajo demanda, no en el cron)
│
├── env_TFG_ETH/                         # entorno virtual (no se sube al repo)
├── .env                                 # claves de API (GEMINI_API_KEY, etc. — no se sube)
├── .gitignore
├── pruebas.ipynb                        # notebook de trabajo
├── README.md                            # este archivo
├── requirements-datos.txt               # dependencias LIGERAS (pipeline diario, sin torch)
├── requirements.txt                     # dependencias COMPLETAS (entorno de desarrollo)
└── validacion_regimenes.txt             # validación histórica de los regímenes del HMM
```

### Rutas clave (para referencia rápida)
| Qué | Ruta |
|-----|------|
| Datos de mercado | `data/csv/raw/df_merged.csv` |
| Regímenes (HMM) | `data/csv/raw/regimenes.csv` |
| Features (LSTM) | `data/csv/processed/df_model.csv` |
| Modelo HMM | `models/hmm_final.pkl` |
| Modelo LSTM | `models/lstm_final.pt` |
| Scalers LSTM | `models/scaler_lstm.pkl` |
| TXT fijos del RAG | `data/txt_rag/fijos/` |
| TXT embebidos del RAG | `data/txt_rag/embebidos/` |

> ⚠️ **Notas sobre la estructura** (importante para evitar errores):
> - Los scripts están en `src/data/`, por lo que `BASE_DIR = Path(__file__).resolve().parent.parent.parent` (3 niveles hasta la raíz).
> - La carpeta de los TXT se llama **`data/txt_rag/`** (no `data/txt/`), con subcarpetas `fijos/` y `embebidos/`.
> - **No existe** un directorio `chroma_db/` todavía (no hay documentos embebidos indexados aún; todos los TXT en uso son fijos).
> - **No existe** un CSV de noticias: las noticias se cargan vía API dentro del notebook del RAG.
> - **No existe** `resultado_eval.json` en este repo.

---

## 9. Estado del proyecto

| Componente | Estado |
|------------|--------|
| Pipeline de datos (df_merged, regímenes, features) | ✅ Funcionando y automatizado (GitHub Actions) |
| HMM | ✅ Entrenado, validado y guardado |
| LSTM | ✅ Entrenada, evaluada y guardada |
| `predecir_lstm.py` (señal bajo demanda) | ✅ Hecho |
| RAG | 🚧 En construcción en este repo (TXT fijos listos; integración de señal LSTM y noticias en curso) |
| ChromaDB / documentos embebidos | ⬜ Pendiente |
| App para la defensa | ⬜ Pendiente |
| Memoria (PDF) | ⬜ Pendiente |

---

## 10. Limitaciones reconocidas

1. **Predicción direccional:** la LSTM no logra una ventaja direccional significativa (~51 %, cercano al azar). Es un resultado esperado y honesto, no un fallo.
2. **Evaluación del RAG:** mientras el HMM y la LSTM se evalúan con métricas cuantitativas claras, la aportación del RAG es difícil de medir objetivamente y queda evaluada de forma **cualitativa** (mediante ejemplos). Como línea futura, se propone evaluar al menos la **calidad de la recuperación** de documentos (precision@k).
3. **Sesgo optimista del RAG:** los textos de contexto tienden a una visión positiva sobre las criptomonedas; se contrarresta con mentalidad *value investing*, pero se reconoce como limitación.

---

## 11. Contribución principal

> **Diseñar y evaluar una arquitectura híbrida que combina modelos probabilísticos de regímenes de mercado (HMM), redes neuronales recurrentes (LSTM) y sistemas de recuperación aumentada por generación (RAG) para apoyar la toma de decisiones en mercados de criptomonedas.**

El valor del trabajo reside en el **rigor metodológico** y en la **orquestación** de información cuantitativa y cualitativa, más que en el porcentaje de acierto de cualquier modelo individual.