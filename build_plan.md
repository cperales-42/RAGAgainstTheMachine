# Plan de acción — Proyecto RAG "RAG against the machine" (42)

> **Objetivo:** construir un pipeline RAG completo sobre el repo vLLM que ingiera, indexe, recupere (BM25/TF-IDF), genere respuestas con `Qwen/Qwen3-0.6B` y mida `recall@k`, cumpliendo el mandatory **al 100%** y los **5 bonuses**, con código limpio, tipado, documentado y sin caídas ante casos degenerados.
>
> **Duración:** 15 días laborables (3 semanas). Cada día = una unidad de trabajo verificable.
> **Filosofía de trabajo:** tú diriges y validas; los agentes de IA escriben **todo** el código y los tests. **No escribes ni una línea.** Aun así, ningún bloque se da por terminado si no puedes explicarlo en voz alta.

---

## 0. Criterios de éxito (contrato del proyecto)

| Requisito | Umbral del subject | Objetivo interno (margen) |
|---|---|---|
| Recall@5 docs | ≥ 80% | ≥ 90% |
| Recall@5 code | ≥ 50% | ≥ 65% |
| Tiempo de indexación | ≤ 5 min corpus completo | ≤ 3 min |
| Throughput retrieval | ≤ 90 s / 200 preguntas | ≤ 30 s |
| Calidad | flake8 + mypy **sin errores** (usar `--strict`) | mypy `--strict` limpio |
| Robustez | nunca crashea con input degenerado | todos los edge cases con test |
| Modelo | `Qwen/Qwen3-0.6B` obligatorio | + embeddings MiniLM (bonus) |
| Gestor | solo `uv sync` (moulinette y reviewer) | `uv.lock` commiteado |
| CLI | Python Fire, `uv run python -m src <cmd>` | docstrings en cada comando |
| Bonus | 5/5 implementados y **demostrables** | 6/5 si se puede |

**Métrica oficial:** `recall@k` calculada por la moulinette. Un source acierta si está **en el mismo `file_path`** y su rango **solapa** con el de referencia (IoU ≥ 0.05). El `file_path` se compara **verbatim**: p. ej. `data/raw/vllm-0.10.1/docs/features/lora.md`.

**Comando oficial de evaluación (referencia):**
```
./moulinette evaluate_student_search_results \
  data/output/search_results/<Scope>/<file>.json \
  data/datasets/AnsweredQuestions/<file>.json \
  --k 10 --max_context_length 2000
```

---

## 1. Cómo vas a usar agentes de IA (sin quedarte fuera del proyecto)

Regla de oro de la Fase de aprendizaje del subject: *"Only use AI-generated content that you fully understand and can take responsibility for."* En este plan **no escribes código**: los agentes implementan todo (código y tests). Tu trabajo es dirigir, revisar, ejecutar, validar y **comprender** lo suficiente para defender el proyecto.

**Reparto de roles:**
- **Tú:** defines el objetivo de cada fase, apruebas las interfaces, lanzas los comandos, revisas los resultados y pides explicaciones hasta entender el sistema.
- **El agente:** escribe el código y los tests, respeta las convenciones del proyecto y explica sus decisiones de diseño.

**Ciclo por cada archivo (bucle de 5 pasos):**
1. **Enuncias el objetivo** en 2-4 líneas de lenguaje natural: qué debe hacer, qué recibe y qué devuelve. No escribes especificaciones formales ni código.
2. **Delegas la implementación** al agente con la plantilla de 1.1 (interfaz exacta, convenciones y tests esperados).
3. **Revisas** el resultado con el checklist de 1.2. Pides explicaciones al agente hasta entender cada línea; si algo cojea, iteras.
4. **Ejecutas** los tests **y** un caso end-to-end real (no solo `pytest`).
5. **Validas y archivas**: si puedes explicar el archivo (a un peer) sin mirar y se comporta como esperabas, queda cerrado; si no, vuelves al paso 2.

### 1.1 Plantilla de prompt para el agente (código)
```
Contexto: proyecto RAG en Python 3.10+. Pipeline: chunking -> índice léxico BM25 -> retrieval -> generación con Qwen3-0.6B.
Tarea: implementa <archivo> con la siguiente interfaz exacta:
  - <clase/función>, firma, tipos de entrada/salida
Restricciones:
  - Type hints completos (mypy --strict), docstrings PEP 257 en inglés, flake8 (max line 79).
  - Sin comentarios inline. Manejo de excepciones con try/except y context managers.
  - No dependas de red ni de estado global oculto.
Entrada/Salida esperada: <ejemplo concreto>
Entrega también los tests en pytest que cubran <casos>.
Al terminar, explica en 5 líneas las decisiones de diseño y los supuestos.
```

### 1.2 Checklist de revisión de código generado
- [ ] ¿Entiendo **cada** línea? ¿Puedo explicar qué hace y por qué sin leerla?
- [ ] ¿Los tipos son correctos y `mypy --strict` pasa?
- [ ] ¿Maneja errores (archivo inexistente, JSON malformado, input vacío)?
- [ ] ¿Usa context managers (`with open(...)`)?
- [ ] ¿No hay números/rutas hardcodeadas que deban venir del CLI?
- [ ] ¿Los índices de carácter (`first_character_index`, `last_character_index`) son correctos y en `[0, len(fichero)]`?
- [ ] ¿Cumple el nombre y la firma exactos pedidos por el subject?

### 1.3 Agentes sugeridos por fase

| Fase | Agente recomendado |
|---|---|
| 0-1 Setup, modelos y E/S | `python-coder` |
| 2 Chunking | `chunking-specialist` |
| 3-4 Tokenizer, BM25, retrieval | `retrieval-specialist` |
| 5-6 Evaluación y tuning | `evaluation-specialist` |
| 7 Generación | `generation-specialist` |
| 8 Lint y tipos | `type-lint-fixer` |
| 9 Bonuses | `bonus-developer` |
| Tests (transversal) | `test-writer` |
| README / documentación | `docs-writer` |
| Revisión de código (transversal) | `code-reviewer` |

---

## 2. Estructura final del repositorio (contrato de layout del subject)

> El subject exige esta estructura para que los scripts de referencia corran sin reestructurar nada.

```
rag/
├── pyproject.toml            # deps + config uv
├── uv.lock                   # commiteado (uv sync)
├── Makefile                  # install, run, debug, clean, lint, lint-strict
├── .gitignore                # ignora data/, __pycache__, .mypy_cache, modelos
├── README.md                 # obligatorio (ver Fase 8)
├── src/                      # módulo ejecutable: python -m src
│   ├── __init__.py
│   ├── __main__.py           # entrypoint Fire
│   ├── cli.py                # comandos: index, search, search_dataset, answer, answer_dataset, evaluate
│   ├── config.py             # constantes, defaults, extensiones soportadas
│   ├── models.py             # modelos pydantic
│   ├── io_utils.py           # carga/guardado JSON con validación y manejo de errores
│   ├── chunking/
│   │   ├── __init__.py       # dispatcher por extensión
│   │   ├── base.py           # Chunk, cálculo de offsets
│   │   ├── python_chunker.py # estrategia código (AST)
│   │   └── markdown_chunker.py  # estrategia markdown/texto
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # tokenización code-aware
│   │   ├── bm25.py           # BM25/TF-IDF
│   │   ├── indexer.py        # recorre corpus, chunk, indexa, persiste
│   │   ├── storage.py        # serialización del índice
│   │   ├── manifest.py       # (bonus 3) hashes por archivo para incremental
│   │   └── embeddings.py     # (bonus 1) vector index MiniLM
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py      # query -> top-k MinimalSource
│   │   └── hybrid.py         # (bonus 2) fusión léxica + semántica
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt.py         # construcción de prompt + presupuesto de tokens
│   │   └── generator.py      # carga Qwen3-0.6B y genera
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── recall.py         # recall@k propio (IoU ≥ 0.05)
│   ├── cache.py              # (bonus 4) cache de índice y queries
│   └── api.py                # (bonus 5) HTTP API local
├── tests/
│   ├── test_models.py
│   ├── test_chunking.py
│   ├── test_tokenizer.py
│   ├── test_bm25.py
│   ├── test_recall.py
│   ├── test_io_utils.py
│   └── test_cli_edge_cases.py
└── data/                     # NO commitear (lo genera el evaluador / tú localmente)
    ├── raw/vllm-0.10.1/               # corpus (¡prefijo exacto!)
    ├── processed/                      # índice
    ├── datasets/{AnsweredQuestions,UnansweredQuestions}/
    └── output/
        ├── search_results/<Scope>/
        └── search_results_and_answer/<Scope>/
```

**Nota crítica de rutas:** el corpus debe estar en `data/raw/vllm-0.10.1/` (el ground truth ya usa ese prefijo). Copia/mueve `vllm-0.10.1/` allí y copia `datasets_public/public/*` a `data/datasets/`. Añade `data/` a `.gitignore`.

---

## 3. Cronograma (3 semanas)

| Día | Fase | Entregable verificable |
|---|---|---|
| 1 | 0. Setup y reconocimiento | Skeleton, uv, Makefile, layout, corpus colocado |
| 2 | 1. Modelos y E/S | `models.py`, `io_utils.py` + tests |
| 3 | 2a. Chunking código | `chunking/base.py`, `python_chunker.py` |
| 4 | 2b. Chunking markdown | `markdown_chunker.py`, dispatcher + tests |
| 5 | 3a. Tokenizer y BM25 | `tokenizer.py`, `bm25.py` + tests |
| 6 | 3b. Indexación | `indexer.py`, `storage.py`, comando `index` |
| 7 | 4. Retrieval | `retriever.py`, `search`, `search_dataset` |
| 8 | 5. Evaluación propia | `recall.py`, `evaluate`, primer baseline con moulinette |
| 9 | 6. Tuning de recall | Umbrales docs ≥80%, code ≥50% superados |
| 10 | 7a. Generación | `prompt.py`, `generator.py`, `answer`, `answer_dataset` |
| 11 | 7b. Calidad de respuestas | Respuestas coherentes y aterrizadas + tests |
| 12 | 8. Robustez y documentación | mypy --strict, flake8, edge cases, README |
| 13 | 9a. Bonuses 1 y 2 | Embeddings + híbrido |
| 14 | 9b. Bonuses 3, 4 y 5 | Incremental + caché + API HTTP |
| 15 | 10. Pulido y defensa | Corrida end-to-end como evaluador + README final |

> **Nota:** este cronograma es un **máximo**, no un mínimo. Como los agentes escriben todo el código, varias fases se solapan y se comprimen; lo que marca el ritmo es tu capacidad de **validar** cada entregable, no de teclearlo.

---

## FASE 0 — Setup y reconocimiento (Día 1)

**Objetivo:** entender el problema a fondo y montar el entorno reproducible.

**Conceptos a dominar antes de tocar código:**
- Las **4 etapas de RAG**: Indexing (organizar para buscar), Retrieving (traer lo relevante), Augmenting (filtrar y meter en la ventana de contexto), Generating (responder).
- El problema *needle-in-a-haystack*: poco retrieval → falta la respuesta; demasiado → ruido que confunde al modelo.
- Diferencia entre *recall* (¿está la aguja entre lo recuperado?) y *precisión* (¿cuánto ruido?).

**Archivos a crear:**
- `pyproject.toml` — metadatos, dependencias y configuración (flake8/mypy).
- `.gitignore` — `data/`, `__pycache__/`, `.mypy_cache/`, `.venv/`, `*.pt`, `*.safetensors`.
- `Makefile` — reglas `install` (`uv sync`), `run`, `debug`, `clean`, `lint`, `lint-strict`.
- `README.md` — primera línea itálica obligatoria `*This project has been created as part of the 42 curriculum by <login>.*`

**Tareas:**
1. `uv init`; añadir deps base: `fire`, `pydantic`, `tqdm`, `numpy`. Dev: `mypy`, `flake8`, `pytest`.
2. Añadir deps de generación: `torch` (CPU), `transformers`.
3. Colocar corpus en `data/raw/vllm-0.10.1/` y datasets en `data/datasets/`.
4. Renombrar el binario de la moulinette (`moulinette-ubuntu` o `-fedora`) a `moulinette` y probar `--help`.
5. Contar el corpus (nº ficheros `.py`/`.md`, MB totales) y anotarlo: condiciona el diseño de indexación.

**Definition of Done:**
- [ ] `uv sync` funciona desde la raíz.
- [ ] `make lint` ejecuta las flags exactas del subject.
- [ ] `python -m src` aún no existe, pero el paquete está creado.
- [ ] Sabes decir en 2 frases qué hace cada etapa de RAG.

**Qué debes entender / verificar:**
- ¿Por qué RAG en vez de reentrenar? — Porque actualizar conocimiento vía fine-tuning es caro/lento; RAG inyecta evidencia externa en el momento de responder.
- ¿Por qué el `file_path` debe ser exacto? — Porque la moulinette compara rutas verbatim.

---

## FASE 1 — Modelos de datos y E/S (Día 2)

**Objetivo:** definir el contrato de datos entre etapas con pydantic.

**Conceptos:**
- **Validación en frontera:** los modelos pydantic validan lo que entra/sale; un JSON malformado se detecta antes de propagar.
- Herencia de modelos (`AnsweredQuestion(UnansweredQuestion)`).
- `default_factory` para `question_id` (UUID) — por qué no `default=` mutable.

**Archivo `src/models.py`:** implementar EXACTAMENTE:
- `MinimalSource(file_path: str, first_character_index: int, last_character_index: int)`
- `UnansweredQuestion(question_id: str = Field(default_factory=...uuid4), question: str)`
- `AnsweredQuestion(UnansweredQuestion)`: `+ sources: List[MinimalSource]`, `+ answer: str`
- `RagDataset(rag_questions: List[AnsweredQuestion | UnansweredQuestion])`
- `MinimalSearchResults(question_id, question, retrieved_sources: List[MinimalSource])`
- `MinimalAnswer(MinimalSearchResults)`: `+ answer: str`
- `StudentSearchResults(search_results: List[MinimalSearchResults], k: int)`
- `StudentSearchResultsAndAnswer(search_results: List[MinimalAnswer], k: int)`
- (Permitido) campos extra: `score`, `is_valid`, `difficulty` con `model_config = ConfigDict(extra="ignore")` para no romper con el ground truth que trae `difficulty`/`is_valid`.

**Archivo `src/io_utils.py`:**
- `load_json(path) -> Any` con manejo de `FileNotFoundError`, `json.JSONDecodeError`.
- `load_rag_dataset(path) -> RagDataset`.
- `save_model(model, path) -> None` creando directorios padres.
- Normalización de rutas posix.

**Definition of Done:**
- [ ] Tests validan errores de validación (índices negativos, tipos).
- [ ] Leer los 4 datasets públicos sin error e imprimir `len(rag_questions)` = 100.
- [ ] `mypy --strict src/models.py src/io_utils.py` limpio.

**Qué debes entender:** ¿por qué pydantic y no dataclasses? → validación + serialización tipada de entrada no confiable.

---

## FASE 2 — Chunking (Días 3-4)

**Objetivo:** dos estrategias de segmentación con offsets de carácter exactos.

**Conceptos:**
- Por qué el chunking importa: decide qué vocabulario queda en el índice. Trocitos grandes → más contexto pero el término se diluye; pequeños → término concentrado pero más fragmentos y ruido.
- **Offsets de carácter**: cada chunk guarda `(file_path, start, end)` sobre el texto original. Toda la evaluación depende de esto.
- **Trampa de offsets con AST:** `col_offset` de Python es offset en **bytes UTF-8**, no en caracteres. Solución robusta: precomputar el índice de inicio de cada **línea** en caracteres y usar `node.lineno`/`node.end_lineno` para los límites (empezar en inicio de línea). Evita bugs con acentos/emojis.
- `--max_chunk_size` por defecto 2000 **caracteres**, nunca superarlo (si no, la salida es inválida entera).

**Archivos:**
- `src/chunking/base.py`:
  - `@dataclass Chunk(file_path, start, end, text, kind)`.
  - `line_start_offsets(text) -> list[int]` (offsets en caracteres).
  - `sliding_window(text, start, end, max_size, overlap) -> iter[Chunk]` para partir bloques grandes.
- `src/chunking/python_chunker.py`:
  - Parsear con `ast.parse`. Recorrer `FunctionDef`, `AsyncFunctionDef`, `ClassDef` (top-level y métodos).
  - Emitir un chunk por unidad lógica (docstring incluida).
  - Si un nodo > `max_chunk_size` → subdividir con ventana deslizante.
  - Fallback si `SyntaxError` → ventana deslizante por líneas.
  - No solapar cobertura duplicada innecesaria; opcional solape pequeño.
- `src/chunking/markdown_chunker.py`:
  - Dividir por encabezados (`#`, `##`, `###`) preservando la jerarquía en el texto del chunk (incluir el título ayuda al matching).
  - Subdividir secciones largas por párrafos, luego ventana deslizante.
  - Aplicable a `.md`, `.txt`, `.rst`.
- `src/chunking/__init__.py`:
  - `chunk_file(path, max_chunk_size) -> list[Chunk]` que despacha por extensión.

**Definition of Done:**
- [ ] Para TODO chunk: `0 ≤ start < end ≤ len(text)` y `end - start ≤ max_chunk_size`.
- [ ] Reconstruir `text[start:end]` coincide con `chunk.text`.
- [ ] Tests con: `.py` normal, `.py` con emojis/acentos, `.py` con `SyntaxError`, `.md` con encabezados y código, fichero vacío.
- [ ] Índice de un fichero grande (p. ej. `vllm/entrypoints/llm.py`) trocea sin superar 2000.

**Qué debes entender:** ¿por qué no usar un `text.split()` por caracteres fijo? → rompe unidades lógicas y pierde contexto; pero sirve como fallback.

---

## FASE 3 — Indexación léxica: tokenizer, BM25, persistencia (Días 5-6)

**Objetivo:** construir y serializar el índice; comando `index`.

**Conceptos clave:**
- **Tokenizer code-aware:** las preguntas citan identificadores (`skip_sampler_cpu_output`, `LLM.embed`) y paráfrasis. Tokeniza separando snake_case y camelCase, conservando el token original y sub-tokens en minúscula; separa por puntuación. No elimines stopwords agresivamente en código.
- **TF-IDF vs BM25:** TF-IDF pondera `tf·idf`; **BM25** añade saturación de término (`k1`) y normalización por longitud del documento (`b`), lo que lo hace superior y robusto a chunks de distinta longitud. Implementaremos BM25 (satisface "al menos uno de los dos").
  - Fórmula: `score(D,Q) = Σ idf(q_i) · (f(q_i,D)·(k1+1)) / (f(q_i,D) + k1·(1 - b + b·|D|/avgdl))`
  - `idf(q) = ln(1 + (N - n(q) + 0.5)/(n(q) + 0.5))`
- **Índice invertido:** mapa `token -> [(chunk_id, tf)]`, más `doc_len[chunk_id]` y `avgdl`. Recuperar una query = iterar solo los postings de sus tokens (rápido, cumple 90 s/200 preguntas).
- **Boost por ruta/nombre de archivo:** indexar tokens del `file_path` y del encabezado del chunk ayuda a preguntas alineadas con nombres de fichero.

**Archivos:**
- `src/indexing/tokenizer.py`: `tokenize(text) -> list[str]` (code-aware).
- `src/indexing/bm25.py`: `BM25Index` con `add_document`, `build`, `search(query, k)`; parámetros `k1`, `b` configurables.
- `src/indexing/storage.py`: `save_index`, `load_index` (pickle determinista o `np.savez` para arrays + JSON para metadata). Incluye versión de formato y config de chunking.
- `src/indexing/indexer.py`: recorre `data/raw/`, filtra extensiones, chunkea, tokeniza, construye índice, guarda en `data/processed/`. `tqdm` mostrando `file/s`, `chunk/s`. Debe terminar < 5 min.
- `src/config.py`: default `max_chunk_size=2000`, extensiones, rutas.
- `src/cli.py` + `src/__main__.py`: comando `index --max_chunk_size <int>`.

**Definition of Done:**
- [ ] `uv run python -m src index --max_chunk_size 2000` genera el índice y muestra progreso.
- [ ] Indexar el corpus completo < 3 min.
- [ ] Recargar el índice y hacer una query manual devuelve resultados sensatos.
- [ ] Tests unitarios de BM25 con un corpus diminuto de scores conocidos.

**Qué debes entender:** explica con tus palabras por qué BM25 saturan el TF y por qué normalizar por longitud. ¿Qué pasa con `b=0` y `b=1`?

---

## FASE 4 — Retrieval (Día 7)

**Objetivo:** de una query a `MinimalSource` ordenados y con rango correcto.

**Conceptos:**
- **Top-k** y rankings; devolución del span exacto del chunk recuperado.
- El span devuelto debe ser **≤ 2000** caracteres (el chunk ya lo garantiza).
- Reutilización de una instancia de índice en memoria (cargar 1 vez) para throughput.

**Archivos:**
- `src/retrieval/retriever.py`:
  - `Retriever(index, chunk_meta)` con `search(query, k) -> list[MinimalSource]`.
  - Manejo de `k=0`, query vacía, query sin tokens válidos → lista vacía sin excepción.
- Ampliar `src/cli.py`:
  - `search <query> --k <int>`: imprime `file_path [start:end]`.
  - `search_dataset --dataset_path <p> --k <int> --save_directory <dir>`: lee dataset, procesa con `tqdm`, escribe `StudentSearchResults` en `<dir>/<nombre_dataset>.json`.

**Definition of Done:**
- [ ] `search "How to configure OpenAI server?" --k 10` devuelve rutas plausibles.
- [ ] `search_dataset` sobre los 4 datasets públicos sin error.
- [ ] Formato de salida válido para la moulinette.
- [ ] 200 preguntas en < 30 s.

**Qué debes entender:** ¿por qué el throughput importa más en batch? ¿cómo afecta `k` al recall pero también al ruido de generación?

---

## FASE 5 — Evaluación propia y primer baseline (Día 8)

**Objetivo:** medir recall@k tú mismo y comparar con la moulinette.

**Conceptos:**
- **Recall@k**: fracción de sources correctos encontrados en el top-k. Con multi-source por pregunta, se promedia.
- **IoU** (Intersection over Union) de rangos: `|A∩B| / |A∪B| ≥ 0.05` cuenta como acierto. Barra baja → no hace falta clavar el span.

**Archivos:**
- `src/evaluation/recall.py`:
  - `overlap_iou(a_start, a_end, b_start, b_end) -> float`.
  - `recall_at_k(student: StudentSearchResults, truth: RagDataset, k) -> dict[str, float]` (1,3,5,10).
  - Emparejar por `question_id`.
- CLI `evaluate --student_search_results_path <p> --dataset_path <p> --k <int>`.

**Tareas:**
1. Correr `search_dataset` + `evaluate` sobre docs y code.
2. Correr la moulinette y comprobar que da cifras coherentes.
3. Anotar baseline (probablemente insuficiente) y preparar tuning.

**Definition of Done:**
- [ ] `evaluate` reproduce aproximadamente las cifras de la moulinette.
- [ ] Baseline documentado para docs y code.

**Qué debes entender:** ¿por qué el recall propio y el de la moulinette deben coincidir? ¿dónde puede haber discrepancias (redondeo, `k`, IoU)?

---

## FASE 6 — Tuning de recall hasta superar umbrales (Día 9)

**Objetivo:** docs ≥ 80% y code ≥ 50% recall@5 (objetivo interno 90/65).

**Palancas de mejora (experimentar de una en una, midiendo):**
1. **Tamaño de chunk**: probar `--max_chunk_size` 800, 1200, 1600, 2000. Chunks menores concentran términos y suben recall en code.
2. **Solape** entre chunks (10-20%) para no cortar la respuesta.
3. **Tokenizer**: incluir sub-tokens de identificadores; eliminar `_`/`#`/`::`; preservar números de versión.
4. **Boost de ruta**: añadir tokens de la ruta y del título markdown con peso.
5. **BM25 params**: `k1 ∈ [1.2, 2.0]`, `b ∈ [0.5, 0.9]`.
6. **Reformulación de query** ligera: normalizar identificadores en la pregunta.
7. **k**: el umbral es recall@5, así que prioriza que los aciertos entren en los 5 primeros.

**Tareas:** construir un pequeño script de experimentos (fuera de `src/`, p. ej. en `tests/` o `scripts/`) que corra `search_dataset` + `evaluate` con distintas configs y tabule resultados.

**Definition of Done:**
- [ ] Recall@5 docs ≥ 80% (objetivo ≥ 90%).
- [ ] Recall@5 code ≥ 50% (objetivo ≥ 65%).
- [ ] Tabla de ablación guardada para el README (Fase 8).

**Qué debes entender:** explica el trade-off entre tamaño de chunk y recall/precisión observado con tus datos.

---

## FASE 7 — Generación de respuestas con Qwen3-0.6B (Días 10-11)

**Objetivo:** comando `answer` y `answer_dataset` con `StudentSearchResultsAndAnswer`.

**Conceptos:**
- **Inferencia local CPU**: cargar `Qwen/Qwen3-0.6B` con `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`). Cachear el modelo entre llamadas.
- **Qwen3 thinking mode**: desactivar el razonamiento (`enable_thinking=False` en `apply_chat_template`) y limpiar ` thinking...` si aparecen; si no, el modelo gasta tokens en razonar y responde peor/rápido.
- **Prompt aterrizado (grounded)**: instruir a responder SOLO con el contexto, citar fuentes y decir "I don't know" si no está. Temperatura 0, `do_sample=False`, `max_new_tokens` acotado.
- **Presupuesto de tokens**: concatenar top-k chunks hasta el límite de contexto (dejar margen para la respuesta). Deduplicar spans solapados.

**Archivos:**
- `src/generation/prompt.py`:
  - `build_prompt(question, sources, chunk_texts, tokenizer, max_context_tokens) -> str`.
  - Formateo tipo chat: system + contexto numerado con `file_path [start:end]` + pregunta.
- `src/generation/generator.py`:
  - `Generator(model_name, device="cpu")` con `generate(question, sources) -> str`.
  - Carga perezosa (lazy) y reutilizable.
- CLI:
  - `answer <query> --k <int>` → imprime respuesta (y sources).
  - `answer_dataset --student_search_results_path <p> --save_directory <dir>`:
    - Relee los `retrieved_sources` del JSON de búsqueda (no rebusca), recupera el texto de cada span leyendo el fichero original, genera y escribe `StudentSearchResultsAndAnswer`.

**Definition of Done:**
- [ ] `answer` responde de forma coherente y aterrizada a 5 preguntas de prueba.
- [ ] `answer_dataset` procesa 100 preguntas mostrando `Processed n of 100`.
- [ ] Manejo de contexto vacío → el modelo no alucina una respuesta firme.
- [ ] Tests con mocks (no cargar el modelo en unit tests).

**Qué debes entender:** ¿por qué es crítico que el contexto sea exactamente los spans recuperados? ¿qué estrategia de prompt reduce alucinaciones?

---

## FASE 8 — Robustez, lint, tipos y README (Día 12)

**Objetivo:** que nada crashee y que el repo sea evaluable tal cual.

**Conceptos:**
- **Manejo de excepciones en frontera**: validar inputs del CLI, nunca dejar trazas.
- `mypy --strict` y flake8 como red de seguridad.

**Tareas:**
1. **Edge cases** (todos con test en `tests/test_cli_edge_cases.py`):
   - query vacía / solo espacios / `k=0` / `k` negativo.
   - `--dataset_path` inexistente o JSON malformado.
   - `--save_directory` inexistente (crear) y sin permisos.
   - dataset vacío (`rag_questions: []`).
   - extensión no soportada / fichero binario / decode error (`errors="replace"`).
   - índice ausente al hacer `search` → mensaje claro, no traceback.
2. Ejecutar `make lint` y `make lint-strict` y arreglar todo.
3. Docstrings PEP 257 en todas las funciones y clases.
4. **README.md completo** con:
   - Primera línea itálica obligatoria con tu(s) login(s).
   - Description, Instructions (install/run), Resources + **uso de IA** (herramientas, tareas, qué partes).
   - System architecture (diagrama del pipeline y cómo interactúan los componentes).
   - Chunking strategy, Retrieval method, Performance analysis (recall@k y tiempos), Design decisions, Challenges faced, Example usage.

**Definition of Done:**
- [ ] `make lint` y `make lint-strict` sin errores.
- [ ] Ningún comando rompe con los edge cases.
- [ ] README cumple todos los apartados exigidos.

**Qué debes entender:** ¿dónde y por qué validas inputs? ¿qué harías ante un JSON con campos extra?

---

## FASE 9 — Bonuses (Días 13-14)

> Solo puntúan si el mandatory está 100% y son **demostrables**. Impleméntalos sin romper el mandatory; cada uno debe tener un testigo de demo.

#1 **Semantic embeddings** (`src/indexing/embeddings.py`)
- **Concepto:** embeddings densos capturan significado, no solo léxico; complementan BM25 ante paráfrasis.
- Modelo: `sentence-transformers/all-MiniLM-L6-v2` (CPU), 384 dims, normalizar vectores.
- Vector index: matriz numpy normalizada (cosine = producto punto) o FAISS `IndexFlatIP` si disponible.
- Persistir en `data/processed/embeddings.npz`.
- **Ojo con el tiempo de indexación:** el mandatory exige ≤5 min. Construye el índice semántico como paso **separado/opcional** (flag `--semantic` o comando `index_embeddings`) con caché, para no violar el límite. Mide.

#2 **Hybrid retrieval** (`src/retrieval/hybrid.py`)
- **Concepto:** fusionar dos rankings. Usa **Reciprocal Rank Fusion (RRF)**: `score(d) = Σ 1/(60 + rank_i(d))`. Robusto y sin normalizar scores.
- `hybrid_search(query, k)` = RRF(BM25 top-N, embeddings top-N) → top-k.

#3 **Incremental indexing** (`src/indexing/manifest.py` + `indexer.py`)
- **Concepto:** no reconstruir todo; detectar cambios por hash (SHA-256) o `mtime+size` por archivo.
- Guardar `manifest.json`: `{file_path: {hash, chunk_ids}}`.
- Al re-indexar: eliminar postings de ficheros borrados/cambiados y añadir solo los nuevos. Verificar que el resultado es idéntico a un indexado completo.

#4 **Caching** (`src/cache.py`)
- **Concepto:** evitar recomputar. Cachear (a) carga del índice/modelo en memoria, (b) resultados de query por `(query, k, config)` en disco (p. ej. `diskcache` o JSON/pickle con hash).
- Cuidado con invalidación: la clave incluye la versión/config del índice.

#5 **Local HTTP API** (`src/api.py`)
- **Concepto:** exponer el motor sin CLI. FastAPI + uvicorn (grupo de deps opcional) con:
  - `GET /search?q=<query>&k=<int>` → `MinimalSearchResults`.
  - `POST /answer` (body: question, k) → `MinimalAnswer`.
  - `GET /health`.
- Reutiliza exactamente `Retriever`/`Generator` (misma lógica, cero duplicación).

**Definition of Done:**
- [ ] Cada bonus tiene una demo reproducible (comando o curl en el README).
- [ ] El mandatory sigue pasando umbrales y lint tras añadirlos.

---

## FASE 10 — Pulido y preparación de defensa (Día 15)

**Tareas:**
1. Correr el flujo **exacto del evaluador**: `index` → `search_dataset` (scoped) → `moulinette evaluate_student_search_results` → `answer_dataset`.
2. Verificar de nuevo umbrales de recall y tiempos (índice, retrieval).
3. Completar README: arquitectura, ablaciones, análisis de rendimiento, decisiones, retos, uso de IA.
4. Preparar el **recode**: poder pedir y validar en minutos la modificación de un parámetro, una función de chunking o un modelo de datos (p. ej. añadir un campo).
5. Repasar el checklist de comprensión de cada fase y explicar el proyecto a un peer.

**Definition of Done:**
- [ ] Evaluación end-to-end verde.
- [ ] Todos los bonus demostrables.
- [ ] Puedes explicar cualquier parte y dirigir/validar una modificación sobre ella.

---

## 4. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Corpus grande → indexación lenta | Falla el límite de 5 min | Tokenizer rápido, índice invertido, evitar embeddings en mandatory |
| Offsets AST en bytes (no chars) | Rangos inválidos → recall 0 | Línea-base de offsets en caracteres; usar `lineno`/`end_lineno` |
| Ficheros `.py` con `SyntaxError` | Crash del indexer | try/except + fallback ventana deslizante |
| `file_path` con prefijo distinto | Nunca casa con ground truth | Indexar desde `data/raw/vllm-0.10.1` y guardar ruta relativa exacta |
| Chunk > 2000 → toda la salida inválida | Recall 0 | Garantizar `end-start ≤ max_chunk_size` con test |
| Modelo pesa GB / sin GPU | Generation lenta o descarga | Descargar con antelación, caché, `max_new_tokens` bajo |
| Embeddings MiniLM lentos en CPU | Rompe límite de indexación | Índice semántico separado + caché |
| JSON con campos extra (`difficulty`) | Error de validación | `extra="ignore"` en modelos |

---

## 5. Apéndice — Conceptos mínimos que debes poder explicar

- **RAG**: recuperar evidencia externa y condicionar la generación; separa el conocimiento del modelo.
- **Chunking**: segmentación en unidades lógicas (AST para código; encabezados para markdown) con offsets exactos.
- **Índice invertido**: `token -> postings`; base de la búsqueda léxica eficiente.
- **TF-IDF**: `tf·idf`; sencillo, sin saturación.
- **BM25**: TF saturado + normalización por longitud; `k1` controla la saturación y `b` la normalización.
- **Recall@k / IoU**: mide si la aguja está en el top-k, con solape mínimo de 0.05.
- **Embeddings**: representación densa semántica; cosine similarity.
- **RRF**: fusión de rankings robusta sin normalización.
- **Grounded generation**: prompt que obliga a responder solo con contexto y a abstenerse si falta.
- **Incremental indexing**: reutilizar trabajo usando hashes por archivo.
- **Caching**: evitar recomputación costosa; cuidar invalidación.

---

## 6. Definición global de "terminado" (mandatory + bonus)

- [ ] Layout exacto exigido y rutas CLI siempre configurables (nunca hardcodeadas).
- [ ] `uv sync` funciona desde la raíz; `uv.lock` commiteado.
- [ ] CLI Fire con `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`.
- [ ] pydantic en los modelos de intercambio; Qwen3-0.6B; tqdm; dos estrategias de chunking; BM25.
- [ ] Recall@5 docs ≥ 80% y code ≥ 50% (ideal 90/65).
- [ ] Indexación ≤ 5 min; retrieval ≤ 90 s/200 preguntas.
- [ ] `make lint` y `make lint-strict` sin errores; sin crashes en edge cases.
- [ ] README completo en inglés con todos los apartados.
- [ ] 5 bonuses implementados y demostrables.
- [ ] Puedes explicar cualquier parte del código y validar una modificación pedida en el recode.