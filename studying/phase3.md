# Phase 3 — Guía de estudio: índice léxico BM25F (`src/indexing/`)

> Fuente: `phases/phase3.md` (spec aprobada). **La fase aún no está
> implementada**: esta guía explica el diseño que se va a construir, con las
> cifras ya medidas sobre el corpus real mediante un prototipo BM25F
> descartable (`/tmp/opencode/bm25f_probe.py`). Cuando la fase esté
> ejecutada, relee también el código real y contrasta.

---

## 0. Mapa de la fase y el flujo de datos

```
data/raw/ (1969 ficheros, 20.2 MB)
   │
   │  Fase 2 (ya hecha): chunk_file → 45.227 Chunk con offsets exactos
   ▼
indexer.index_corpus(IndexConfig)
   │
   ├─ Fase A "Chunking":  por fichero  (tqdm, 1969 it.)
   │     read_file_text + chunk_file + validate_chunks
   │
   └─ Fase B "Tokenizing": por chunk  (tqdm, 45.227 it.)
         tokenize(body) + tokenize(header) + tokenize(path)
         BM25Index.add_document(doc_id, campos)
   │
   ▼
index.build()  → arrays CSR en memoria
   │
   ▼
storage.save_index(...)
   ├── data/processed/index.npz        (~3 MB, arrays numpy)
   └── data/processed/index_meta.json  (~10 MB, pydantic)
```

Archivos que crea esta fase:

```
src/
├── config.py              # defaults + IndexConfig (NUEVO)
├── models.py              # + IndexedChunk (solo aditivo)
├── cli.py                 # class Rag con el comando index (NUEVO)
├── __main__.py            # fire.Fire(Rag) (NUEVO)
└── indexing/
    ├── __init__.py        # re-exports
    ├── tokenizer.py       # tokenize(): code-aware
    ├── bm25.py            # BM25Index: BM25F + arrays CSR
    ├── storage.py         # IndexMeta, save_index, load_index
    └── indexer.py         # index_corpus(): walk, chunk, index, persist
```

**En una frase:** la Fase 3 convierte los chunks de la Fase 2 en un índice
invertido consultable en milisegundos, lo persiste de forma determinista y
expone el comando `index` de la CLI.

---

## 1. ¿Qué es un índice léxico y para qué existe?

Un modelo BM25 **no entiende** preguntas: compara conjuntos de tokens. El
índice léxico es la estructura que hace esa comparación barata:

- **Índice invertido:** `token → lista de (doc_id, tf)`. Sin él, responder a
  una pregunta exigiría recorrer los 45.227 chunks enteros sumando
  coincidencias (inaceptable para 200 preguntas en < 90 s).
- **Postings:** cada entrada de la lista es un *posting*
  `(documento, frecuencia del término en ese documento)`. El número total de
  postings del corpus real es ~2.0 millones en los tres campos.
- **¿Por qué léxico y no semántico?** El subject exige TF-IDF **o** BM25 en
  la parte obligatoria (VI.1); los embeddings densos son el bonus 1, una capa
  aparte que se fusionará con este ranking en la Fase 4/bonus 2.

La calidad del retrieval depende de **qué tokens metes al indexar**. Una
pregunta puede parafrasear o citar identificadores literales
(`skip_sampler_cpu_output`): si el vocabulario indexado no los conserva, la
coincidencia es imposible. De ahí que el tokenizer sea *code-aware*.

---

## 2. Tokenizer *code-aware* (`src/indexing/tokenizer.py`)

### 2.1 El algoritmo (normativo, los tests fijan las listas exactas)

```
1. partir el texto por runs de \W  (re.UNICODE; conserva letras acentuadas
   y el guion bajo _)
2. emitir cada pieza en minúsculas
3. subdividir la pieza en camelCase (_CAMEL_SEP) y luego por "_"
   → emitir cada subtoken en minúsculas, SOLO si != de la pieza completa
```

Ejemplos que los tests deben repetir al pie de la letra:

| entrada | salida |
|---|---|
| `""` / `"-- ++ "` | `[]` |
| `"Hello World"` | `["hello", "world"]` |
| `"skip_sampler_cpu_output"` | `["skip_sampler_cpu_output", "skip", "sampler", "cpu", "output"]` |
| `"getUserName"` | `["getusername", "get", "user", "name"]` |
| `"LLMEngine"` | `["llmengine", "llm", "engine"]` |
| `"parseHTTPResponse_data"` | `["parsehttpresponse_data", "parse", "http", "response", "data"]` |
| `"v0.10.1"` | `["v0", "10", "1"]` |
| `"café naïve"` | `["café", "naïve"]` |
| `"foo foo FOO"` | `["foo", "foo", "foo"]` |

### 2.2 ¿Qué NO hace y por qué?

Ni stopwords, ni stemming, ni deduplicado, ni filtro de longitud:

- **Sin stopwords:** `for`, `is`, `import`, `self` son "ruido" en inglés pero
  **estructura** en Python; el corpus real los tiene en >9k chunks cada uno.
  Quitarlos destruiría citas literales de código.
- **Sin deduplicado:** la frecuencia de término (tf) es señal para BM25; si
  `hello` aparece 3 veces, el índice debe saberlo.
- **Minúsculas siempre:** `FOO` y `foo` deben casar; además garantiza
  `t == t.lower()` para todo token de salida.

### 2.3 La regla de oro: una sola función

**La MISMA `tokenize` procesa documentos y consultas.** Cualquier asimetría
(minúsculas solo al indexar, subtokens solo en la query...) destruye el
recall en silencio: los tokens de la pregunta y los del índice dejarían de
vivir en el mismo vocabulario.

---

## 3. BM25: la fórmula y sus dos perillas

BM25 puntúa la relevancia de un documento `d` para una consulta `Q` como la
suma, sobre los tokens únicos de `Q`, de:

```
score(t, d) = idf(t) · (k1 + 1) · tf / (k1 + tf)

tf, en nuestro caso, es el tf_hat agregado de BM25F (sección 4)
```

### 3.1 `idf` suavizado (siempre positivo)

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

- `N` = número de documentos; `df(t)` = documentos que contienen `t` en
  **algún** campo.
- El idf clásico `ln(N/df)` se vuelve **negativo** para términos muy
  comunes: los documentos que coinciden puntuarían POR DEBAJO de los que no.
  El suavizado de la fórmula anterior es siempre `> 0` y elimina ese efecto
  (test 36: un término presente en TODOS los documentos sigue devolviendo
  todos con `score > 0`).

### 3.2 Saturación (`k1`) y normalización por longitud (`b`)

| parámetro | papel | extremos |
|---|---|---|
| `k1` (default 1.5) | satura la tf: pasar de 1 a 3 apariciones del mismo token aporta cada vez menos | `k1→0` ≈ booleano (o está o no); `k1→∞` ≈ tf lineal |
| `b` (default 0.75) | normaliza por longitud de documento: chunks largos "diluyen" su tf | `b=0` ignora longitud; `b=1` normaliza al completo (`len/avgdl`) |

¿Por qué normalizar longitud? Un chunk de 2000 caracteres contiene casi
cualquier token de pasada; el mismo token dentro de un chunk de 40 caracteres
es evidencia mucho más fuerte. Sin `b`, los ficheros grandes ganan siempre.

Ejemplo de saturación (test 32): con `k1=1.5` y `b=0`, duplicar la tf de 2 a
4 **no** dobla el score: lo multiplica por menos de 2 (sublineal a propósito).

---

## 4. BM25F: la mejora aprobada sobre el plan

### 4.1 El problema con el "boost" del `build_plan.md`

El plan original sugiere **duplicar** los tokens de la ruta y la cabecera
dentro del texto del chunk para que pesen más. Medido y descartado:

1. **La saturación anula las repeticiones ficticias.** Repetir un token 3
   veces para "darle peso 3" no triplica su score: con `k1=1.5`, el factor
   pasa solo de 1.00 a 1.67. El "peso" quedaría acoplado a `k1` y sería
   ingobernable en el tuning (Fase 6).
2. **La normalización debe ser por campo.** Un acierto en una cabecera de 10
   tokens es evidencia mucho más fuerte que el mismo token en un cuerpo de
   600. Mezclar los campos en una bolsa única los trata igual; BM25F
   normaliza cada campo con **su propia** longitud media.
3. **Duplicar contamina las estadísticas.** Los tokens inyectados inflan la
   longitud del documento (el `b` ve documentos más largos de lo que son) y
   distorsionan la tf del campo cuerpo.
4. **Tuning ortogonal:** `k1` (saturación), `b` (longitud) y
   `w_body`/`w_header`/`w_path` (tipo de evidencia) se mueven **uno a uno**.
5. **Sigue siendo BM25** (familia Okapi, mismo idf, misma saturación): el
   mandatory de VI.1 se cumple literalmente.

### 4.2 Los tres campos de cada documento

| campo | contenido | peso default |
|---|---|---|
| `body` | `tokenize(chunk.text)` — el texto completo del chunk | 1.0 |
| `header` | `tokenize(primera línea no vacía del chunk)` | 2.0 |
| `path` | `tokenize(ruta relativa al corpus)` — sin el prefijo `data/raw/` constante | 2.0 |

- El campo `header` **recupera en tiempo de indexado lo que la Fase 2 decidió
  no meter en el texto del chunk** (los títulos de sección markdown, la
  firma de la función): coherencia entre fases, sin romper el invariante
  `text[start:end] == chunk.text`.
- En un chunk markdown, la primera línea no vacía suele ser `## título`; en
  uno Python, el decorador o la firma.

### 4.3 La fórmula BM25F (normativa)

```
norm_f(d)   = 1 - b + b · len_f(d) / avgdl_f     # 1.0 si avgdl_f <= 0
tf_hat(t,d) = Σ_f  w_f · tf_f(t,d) / norm_f(d)
score(t,d)  = idf(t) · tf_hat(t,d) · (k1+1) / (k1 + tf_hat(t,d))
score(Q,d)  = Σ_{t únicos de tokenize(Q)} score(t, d)
```

Claves a retener:

- La **saturación se aplica UNA sola vez, sobre `tf_hat` ya agregado**. Eso
  es lo que hace meaningful los pesos de campo (motivo 1 de 4.1).
- La **query deduplica** sus tokens (`"openai openai server"` →
  `["openai", "server"]`): si no, contar dos veces el mismo término duplicaría
  su contribución.
- Solo se devuelven documentos con `score > 0` (comparten al menos un token
  con la query); el resto rellenaría el top-k de ruido y empeoraría la
  generación de la Fase 7.
- Orden: `(-score, doc_id)` — los empates se rompen por `doc_id` ascendente
  para que dos corridas idénticas produzcan rankings bitwise idénticos.

---

## 5. Estructuras de datos: dicts → CSR → `.npz` (sin pickle)

### 5.1 Acumulación y `build()`

- Durante `add_document` se llena, por campo, un dict
  `token → {doc_id: tf}` (medido: 4.9 s y pico de 191 MB para el corpus
  entero — código simple y holgado dentro del presupuesto).
- `build()` congela los dicts en arrays **CSR** (compressed sparse row) sobre
  el vocabulario ordenado lexicográficamente (determinismo): una fila
  contigua de postings por token.

| clave | dtype | significado |
|---|---|---|
| `vocab` | `str_` | tokens únicos ordenados (V ~59k) |
| `indptr` | `int64` | la fila `i` ocupa `[indptr[i], indptr[i+1])` |
| `docs` | `int32` | doc ids ascendentes por fila (P ~2.0M) |
| `tf_body` / `tf_header` / `tf_path` | `int32` | tf de cada posting por campo |
| `len_body` / `len_header` / `len_path` | `int64` | longitud de cada campo por documento |
| `idf` | `float64` | idf precalculado por fila de vocabulario |

`search` recorre solo los postings de los tokens de la query, ya vectorizado
con numpy. Medido: 5 consultas en 0.096 s → ~3.9 s proyectadas para 200
(límite duro: 90 s; objetivo interno: 30 s).

### 5.2 Persistencia: `index.npz` + `index_meta.json`

- **`index.npz`**: exactamente las diez arrays anteriores, escritas con
  `savez_compressed` (~3 MB) y leídas con `allow_pickle=False`.
  ¿Por qué **nunca pickle**? Pickle ejecuta código al cargar (puerta para
  datos corruptos o maliciosos), trae trampas de versión de protocolo y no
  aporta nada aquí: numpy serializa estos arrays de forma nativa. Load
  medido: 0.04 s.
- **`index_meta.json`**: modelo pydantic `IndexMeta` con
  `format_version`, `config` (el `IndexConfig` completo), `chunks`
  (45.227 `IndexedChunk`) y `avgdl`. Se escribe con el `save_model` de la
  Fase 1 → directorios padres y errores limpios gratis.

### 5.3 Los invariantes que atan las fases

- **`chunks[doc_id]` describe al documento `doc_id`.** El índice numera
  documentos 0..N-1; el registro `i` de la lista es LA localización del
  documento `i`. Las Fases 4–7 dependen de esto: `hit.doc_id → chunks[doc_id]
  → MinimalSource`.
- **`format_version`**: si cambia el formato de los arrays, se incrementa y
  el loader rechaza mezclar versiones con un mensaje que nombra ambas.
- `from_bundle` valida al cargar: las diez claves presentes, `indptr` coherente
  con `vocab`/`docs`, `docs` entera, `avgdl` cubriendo exactamente los tres
  campos. Un npz corrupto falla **ahí**, no como recall inexplicable luego.
- El **texto de los chunks no se persiste**: la Fase 7 relee
  `text[first:last]` del fichero original. El corpus es la única fuente de
  verdad del contenido.

`IndexedChunk` (nueva clase de `src/models.py`) extiende `MinimalSource` con
`kind: Literal["python", "markdown", "sliding"]` — permite diagnosticar en la
Fase 6 qué chunks de sliding-window (cortes arbitrarios) fallan. Convención
heredada de la Fase 1: `last_character_index` es el fin **exclusivo**
(semántica de slicing de Python).

---

## 6. `file_path` verbatim: la regla de rutas

La moulinette compara rutas **literalmente**
(`data/raw/vllm-0.10.1/docs/features/lora.md`). Un prefijo distinto = recall
cero. El indexer es estricto:

1. Raíz del proyecto = `Path.cwd().resolve()` (la CLI se ejecuta siempre en
   la raíz; el código nunca cambia el directorio de trabajo).
2. Si `corpus_path` es relativo, se resuelve contra la raíz; si falta o no es
   directorio → `ValueError` nombrando el valor configurado.
3. El corpus **debe estar estrictamente dentro** de la raíz del proyecto: un
   corpus fuera no puede producir rutas relativas verbatim, y escribir
   rutas absolutas o truncadas en silencio haría fallar TODA la evaluación.
   `corpus_path="."` también se rechaza (borraría el prefijo `data/raw/`).
4. `file_path` grabado = `to_posix_path(relativo_a_la_raíz)` → exactamente
   `data/raw/vllm-0.10.1/...`.
5. El campo `path` del documento usa la ruta **relativa al corpus** (sin el
   `data/raw/` compartido por todos). Los tokens aún constantes (p. ej. la
   versión `vllm-0.10.1`) tienen `df == N` → `idf ~ 0` → no cuestan nada en
   tiempo de query.

---

## 7. El indexer en dos fases (con las barras tqdm del subject)

El subject muestra dos barras, `Chunking:` (por ficheros) y `Tokenizing:`
(por chunks). `index_corpus` las implementa literalmente:

**Fase A — Chunking** (`tqdm`, `unit="file"`, 1969 ficheros, medido 3.8 s):

```python
text = read_file_text(file_path)
chunks = chunk_file(file_path, config.max_chunk_size, config.overlap)
validate_chunks(text, chunks, config.max_chunk_size)
```

Se guardan pares `(file_path, chunks)` en memoria (~25 MB). `validate_chunks`
vuelve a correr aquí (defensa en profundidad de la Fase 2): nada que viole un
invariante puede llegar a la persistencia.

**Fase B — Tokenizing** (`tqdm`, `unit="chunk"`, 45.227 chunks): para cada
chunk se construyen los tres campos (sección 4.2), se llama
`index.add_document(doc_id, fields)` y se acumula el `IndexedChunk`
correspondiente. Al final: `index.build()`, `save_index(...)` y un
`IndexStats` (pydantic) con los contadores y el tiempo.

Sin `try/except` dentro de `index_corpus`: cada capa inferior ya convierte sus
fallos a `ValueError` (lectura, chunking, config, storage) y la CLI (siguiente
sección) los transforma en mensajes limpios. Los bugs de programación deben
seguir estallando con traceback **durante el desarrollo**.

---

## 8. Config, CLI Fire y el pyproject

### 8.1 `src/config.py` — única fuente de defaults

Todos los `DEFAULT_*` (corpus, processed, chunk size 2000, overlap 0, k1,
b, pesos) viven aquí. Ni `cli.py` ni `indexing/*` pueden repetir un literal
de default: los importan. `SUPPORTED_EXTENSIONS` se **re-exporta** desde
`src.chunking` (ya existe; no se redefine).

`IndexConfig` es un pydantic `BaseModel` validado **en la frontera**: `k1 > 0`,
`0 <= b <= 1`, pesos `>= 0`, `max_chunk_size >= 1` y un validador de modelo
que rechaza `overlap >= max_chunk_size` (la misma regla que `sliding_window`
de la Fase 2, pero capturada ANTES de tocar un fichero). La CLI construye
este objeto desde argv no confiable, y `storage.py` lo relee del JSON luego:
un objeto validado reemplaza a comprobaciones ad-hoc por todas partes.

### 8.2 `src/cli.py` + `src/__main__.py`

- Clase `Rag` con método `index(...)`; cada parámetro del CLI tiene de
  default la constante de `config.py`. Fire mapea
  `uv run python -m src index --max_chunk_size 2000` a
  `Rag().index(max_chunk_size=2000, ...)`.
- **Contrato de errores** (subject VI.6: un traceback no manejado durante la
  revisión = no funcional):

```python
try:
    config = IndexConfig(...)
    stats = index_corpus(config)
except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1) from None
```

  `pydantic.ValidationError` **subclasea** `ValueError`, así que
  `--max_chunk_size 0`, `overlap >= max` o `b=2` comparten el mismo camino
  limpio: mensaje en stderr, exit 1, jamás traceback. Solo se captura
  `ValueError`: los bugs reales conservan su traceback en desarrollo.
- `__main__.py` llama `fire.Fire(Rag)` — esto además hace meaningful
  `make run` y `make debug`.

### 8.3 Los dos cambios de `pyproject.toml` (primera tarea)

`mypy --strict` hoy falla con `import fire` y `import tqdm` ("missing library
stubs"). Arreglo mínimo y acotado:

1. `uv add --dev types-tqdm` (stubs comunitarios; el runtime no cambia).
2. `fire` no tiene stubs en ningún sitio → override de mypy **solo** para ese
   módulo:

   ```toml
   [[tool.mypy.overrides]]
   module = ["fire"]
   ignore_missing_imports = true
   ```

`make lint-strict` debe seguir en verde antes de escribir código de la Fase 3.
Es una decisión de diseño documentable para el README: el override toca un
único módulo de terceros sin tipos y no afloja la comprobación de `src/`.

---

## 9. Números medidos (prototipo, 2026-10-19)

Sirven como presupuesto y como "reporta tu desviación" de la verificación:

| hecho | valor |
|---|---|
| ficheros soportados | 1969 (1763 `.py`, 178 `.md`, 28 `.txt`) |
| chunks (`max=2000`, `overlap=0`) | 45.227 (43.506 python + 1.721 md/txt) |
| chunking del corpus completo | 3.8 s |
| BM25F build (tokenizar + postings + CSR) | 9.1 s wall, 191 MB pico RSS |
| vocabulario / postings | ~59k tokens / ~2.0M (3 campos) |
| npz guardado / cargado | 0.6 s, ~3 MB / 0.04 s |
| 5 queries | 0.096 s → ~3.9 s proyectadas para 200 (límite 90 s) |
| top-1 correcto en 5 consultas de prueba | `openai_compatible_server.md`, `lora.md`, `parallelism_scaling.md`, `structured_output/`, `sampler.py` |

Total del comando `index` proyectado: **~10 s** para un límite duro de
300 s — margen enorme (ese es el sentido de medir antes de implementar).

---

## 10. Autoevaluación — preguntas para defenderte

1. **¿Por qué el tokenizer no elimina stopwords ni hace stemming?** —
   Porque `for`/`is`/`import`/`self` son estructura en Python (estarían en
   >9k chunks), las preguntas citan identificadores literales, y la tf sin
   deduplicar es señal para BM25. Cualquier filtrado asimétrico destruye
   recall en silencio.
2. **¿Qué produce `tokenize("skip_sampler_cpu_output")` y por qué el token
   completo va primero?** —
   `["skip_sampler_cpu_output", "skip", "sampler", "cpu", "output"]`. El
   completo primero permite matchear la cita literal de la query; los
   subtokens permiten matchear preguntas que solo nombran una parte.
3. **¿Qué dos perillas tiene BM25 y qué controla cada una?** — `k1`
   (saturación de la frecuencia: cuánto importa repetir un término) y `b`
   (normalización por longitud: cuánto penaliza chunks largos). Con `b=0` la
   longitud se ignora; con `b=1` se normaliza al completo.
4. **¿Por qué el idf suavizado y no el clásico `ln(N/df)`?** — El clásico se
   vuelve negativo para términos muy comunes: los documentos que coinciden
   quedarían por debajo de los que no. El suavizado `ln(1 + (N-df+0.5)/(df+0.5))`
   es siempre positivo.
5. **¿Por qué BM25F en vez de duplicar tokens de la ruta en el texto?** —
   La saturación (`k1`) anula las repeticiones ficticias (de "peso 3" solo
   queda 1.67×), la longitud debe normalizarse **por campo**, y duplicar
   contamina la tf del cuerpo y el cálculo de `b`. BM25F deja los tres
   ejes (k1, b, pesos) ortogonales para el tuning de la Fase 6.
6. **¿Dónde satura la frecuencia en BM25F y por qué ese orden?** — La
   saturación se aplica **una sola vez**, sobre `tf_hat` ya agregado
   (`Σ w_f·tf_f/norm_f`). Si saturara por campo antes de combinar, los
   pesos `w_f` no tendrían un efecto lineal e interpretable.
7. **¿Qué campos tiene cada documento, cómo se calcula el `header` y por
   qué?** — `body` (texto del chunk), `header` (primera línea no vacía) y
   `path` (ruta relativa al corpus). El `header` recupera en indexado lo que
   la Fase 2 decidió no anteponer al texto (títulos/firmas) sin romper
   `text[start:end] == chunk.text`.
8. **¿Qué invariant liga el índice con la lista de chunks y quién depende de
   él?** — `chunks[doc_id]` describe al documento `doc_id` (numeración
   0..N-1). Las Fases 4–7 lo usan para traducir `hit.doc_id` a
   `MinimalSource` y, en la Fase 7, a relectura del span original.
9. **¿Por qué `.npz` con `allow_pickle=False` y nunca pickle?** — Pickle
   ejecuta código al cargar, trae trampas de versión de protocolo y no
   aporta: numpy serializa los arrays nativamente. El loader valida además
   las diez claves y las longitudes antes de devolver el índice.
10. **¿Por qué el corpus debe estar estrictamente dentro del CWD?** — Porque
    el `file_path` debe grabarse verbatim `data/raw/vllm-0.10.1/...` para que
    la moulinette (que compara rutas literales) acierte; un corpus fuera no
    admite ruta relativa correcta, y `corpus_path="."` borraría el prefijo.
11. **Cómo maneja la CLI un `--max_chunk_size 0` sin crashear?** — El valor
    invalida `IndexConfig` al construirlo; `ValidationError` subclasea
    `ValueError`; el `except ValueError` imprime `error: ...` en stderr y
    hace `SystemExit(1)`. Nunca traceback.
12. **Si indexas con una `tokenize` y buscas con otra, ¿qué pasa?** — Los
    tokens de la pregunta y los del índice dejan de compartir vocabulario;
    el recall cae en silencio (no hay error, simplemente no hay coincidencias
    léxicas). Por eso `tokenize` es una única función usada en ambos lados.
