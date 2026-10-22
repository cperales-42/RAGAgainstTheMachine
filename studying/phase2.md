# Phase 2 — Guía de estudio: `src/chunking/`

> Fuente: `phases/phase2.md` (fase ejecutada y verificada) + código real de
> `src/chunking/{__init__,base,python_chunker,markdown_chunker}.py`.

---

## 0. Mapa del paquete y los 3 invariantes

```
src/chunking/
├── __init__.py         # despachador: chunk_file + SUPPORTED_EXTENSIONS
├── base.py             # Chunk, line_start_offsets, read_file_text,
│                       #   sliding_window, validate_chunks
├── python_chunker.py   # estrategia AST      → kind "python" | "sliding"
└── markdown_chunker.py # estrategia cabeceras → kind "markdown" | "sliding"
```

Los **3 invariantes** que TODO chunk de TODOA estrategia debe cumplir:

```
1) 0 <= start < end <= len(text)        # el chunk es una rebanada válida
2) end - start <= max_chunk_size        # 2000: la moulinette invalida TODA
                                        #   la salida si algún source lo supera
3) text[start:end] == chunk.text        # lo indexado == lo que se re-leerá
```

`kind` solo puede ser `"python"`, `"markdown"` o `"sliding"` (`CHUNK_KINDS`).

---

## 1. `__init__.py` — `chunk_file`: ¿cuándo se usa y para qué?

### Qué hace

`chunk_file(path, max_chunk_size, overlap=0) -> List[Chunk]` es la **única
puerta de entrada** al paquete. En tres pasos:

1. Comprueba la extensión PRIMERO (lo más barato y el error más claro):
   `Path(path).suffix.lower()` debe estar en `SUPPORTED_EXTENSIONS`
   `(".py", ".md", ".txt", ".rst")` — si no → `ValueError` que nombra la
   extensión y el conjunto soportado.
2. Normaliza la ruta a POSIX (`to_posix_path`) y lee el fichero
   (`read_file_text`: UTF-8 con `errors="replace"`, context manager).
3. Despacha: `.py` → `chunk_python`; `.md`/`.txt`/`.rst` → `chunk_markdown`.

El llamador **no tiene que saber** qué estrategia le toca: le da un fichero y
recibe chunks validados cuyo `file_path` ya está en formato POSIX.

### ¿Cuándo se usa?

| Momento | Quién lo llama | Para qué |
|---|---|---|
| **Ahora (Fase 2)** | `tests/test_chunking.py` (casos 41–46) y el smoke test de la sección 9 del plan | verificar el despacho |
| **Fase 3 (`indexer` + comando `index`)** | el indexer al recorrer `data/raw/` | **uso principal**: chunkear los 1763 `.py` + 206 `.md/.txt/.rst` del corpus y construir el índice |
| **Fase 7 (generación)** | no llama a `chunk_file`, pero lee `text[start:end]` con la MISMA política de lectura (`read_file_text`) | reconstruir el contexto exacto de cada source recuperado → coherencia índice/generación |

Regla crítica que repite el docstring: el llamador **debe pasar rutas relativas
a la raíz del proyecto** (`data/raw/vllm-0.10.1/docs/features/lora.md`), porque
la moulinette compara `file_path` **verbatim**. El indexer (Fase 3) es el
responsable de "rootear" cada ruta antes de llamar.

Detalle: `SUPPORTED_EXTENSIONS` es la **única fuente de verdad** de extensiones;
`src/config.py` (Fase 3) debe importarla, no redefinirla.

```python
from src.chunking import chunk_file
chunks = chunk_file("data/raw/vllm-0.10.1/vllm/x.py", max_chunk_size=2000)
```

---

## 2. `base.py` — ¿por qué los offsets importan tanto y tantas comprobaciones?

### 2.1 Por qué los offsets son el corazón de todo el proyecto

Un chunk **no es solo texto**: es una rebanada con "dirección postal"
`(file_path, start, end)`. Esos tres números viajan hasta el final del
pipeline:

1. **Retrieval (VI.2):** la salida de `search` es
   `MinimalSource(file_path, first_character_index, last_character_index)` —
   son literalmente el `start`/`end` del chunk. El retrieval no devuelve
   "parte del texto", devuelve **dónde está**.
2. **Moulinette / recall@k:** un source cuenta como acertado solo si el
   `file_path` coincide Y los rangos se solapan con **IoU ≥ 0.05**. Offsets
   desplazados → IoU 0 → recall 0 **aunque el contenido sea perfecto**.
3. **Generación (Fase 7):** `answer_dataset` relee `text[start:end]` del
   fichero ORIGINAL para montar el prompt. Offset equivocado → contexto
   equivocado → el modelo alucina o responde de otro tema.
4. **Límite de 2000 caracteres:** la moulinette corre con
   `--max_context_length 2000` y rechaza **toda la salida de la corrida** si
   un solo source lo supera → `end - start <= max_chunk_size` es una regla
   dura, no una preferencia.
5. **Invariante de rebanada (`text[start:end] == chunk.text`):** garantiza que
   *lo indexado es exactamente lo que se re-leerá después*. Por eso la Fase 2
   tomó la decisión 5.3 de **NO** anteponer títulos de cabeceras al texto del
   chunk (como sugería `build_plan.md`): sintetizar texto rompería la
   rebanada. La jerarquía se recupera luego en tiempo de indexado (Fase 3,
   boost de tokens de cabecera/ruta).

**En una frase:** los offsets son el contrato entre las 4 etapas de RAG —
indexar con offsets incorrectos envenena retrieval, evaluación y generación a
la vez.

### 2.2 Por qué tantas comprobaciones (defensa en profundidad)

Cada nivel valida lo que puede, en el momento más barato posible:

| Dónde | Qué valida | Cuándo falla |
|---|---|---|
| `Chunk.__post_init__` | `file_path` no vacío, `start >= 0`, `end > start`, `kind` válido | **al crear** el chunk, mensaje que nombra el campo culpable |
| `_validate_window_params` | `max_chunk_size > 0`, `0 <= overlap < max_chunk_size` | **antes** de procesar nada |
| `validate_chunks` (final de cada strategy) | rebanada, rango, tamaño ≤ max, `start` estrictamente creciente (orden de fichero) | cuando la lista ya está construida, mensaje con índice del chunk |

Motivos concretos:

- **Evitar un bucle infinito real:** en `sliding_window`,
  `cursor = chunk_end - overlap`. Si `overlap >= max_chunk_size`, el cursor
  **no avanza** → bucle infinito. Con la validación, el paso es
  `max_chunk_size - overlap >= 1` → progreso garantizado (por eso
  `overlap == max - 1` es válido: avanza 1 carácter por ventana).
- **Fallar barato y local:** un `ValueError` en la creación dice exactamente
  qué campo está mal; sin validación, el error aparecería mucho más abajo
  (en la moulinette) y sería intratable.
- **`ValueError` nunca `AssertionError`:** los `assert` se desactivan con
  `python -O`; la validación de contratos debe ser permanente.
- **Nunca crashear (regla V.1 del subject):** `read_file_text` envuelve todo
  `OSError`/`UnicodeError` en `ValueError` con la ruta en el mensaje; bytes
  inválidos se reemplazan (`errors="replace"`), no estallan.
- **`validate_chunks` es la defensa en profundidad de la regla de 2000:**
  `chunk_python` y `chunk_markdown` lo llaman antes de devolver, y el indexer
  (Fase 3) y el retriever (Fase 4) lo llamarán antes de persistir/devolver.
  Un chunk gigante o inconsistente **nunca puede silenciosamente envenenar una
  corrida de evaluación entera**.

### 2.3 El papel de `line_start_offsets`

Convierte **líneas** (1-based, como las da el AST: `node.lineno`) en
**offsets de caracteres** (0-based): `offsets[i]` = posición del primer
carácter de la línea `i`. Se usa para cortar siempre **al inicio de línea**
(invariante de estilo: ningún chunk empieza a mitad de línea).

Evita la trampa clásica: `col_offset` de CPython está en **bytes UTF-8**, no
en caracteres — con `# café 🚀` los offsets se desplazarían y el test 29
(slice consistency) fallaría. Cortar por `lineno` + `line_start_offsets` es
siempre correcto en caracteres.

---

## 3. `python_chunker.py` — estrategia AST, fallback y `sliding_window`

### 3.1 ¿Qué es la estrategia AST y cómo la estoy aplicando?

**AST** = *Abstract Syntax Tree*. `ast.parse(text)` convierte el código Python
en un árbol de nodos (`Module`, `FunctionDef`, `ClassDef`, `Import`...).
En vez de cortar a ciegas por caracteres, cortamos por **unidades lógicas del
lenguaje**: un import, una función con su docstring, una clase con sus
miembros... El índice gana vocabulario coherente y el retrieval recupera
"un tema entero" en vez de un trozo arbitrario.

Cómo se aplica aquí — tres reglas encadenadas:

**Regla P1 — partición del módulo** (`_partition_module`):

- Recorre **solo** `tree.body` (sentencias de primer nivel). Los `def`
  anidados dentro de funciones **no** se promueven → cero cobertura
  duplicada.
- Para cada nodo calcula `after(node)` = offset de inicio de la línea
  **siguiente** a la de su final (`node.end_lineno` vía
  `line_start_offsets`; si esa línea no existe → `len(text)`).
- Los spans son consecutivos: `[0, after(n0))`, `[after(n0), after(n1))`,
  ..., `[after(n_{k-1}), len(text))` → **cobertura exacta de `[0, len(text))`,
  sin huecos ni solapes**.

Consecuencias (todas intencionadas):

- docstring del módulo, comentarios iniciales e imports → chunks tempranos;
- **comentario entre dos nodos → se adhiere al nodo SIGUIENTE** (un comentario
  encima de `def foo` suele documentar a `foo`);
- contenido tras el último nodo → último chunk;
- **decoradores dentro del span del nodo decorado**
  (`_statement_start_line` usa la línea del primer `@decorator`, para no
  orfanar el decorador de su función).

**Regla E1 — emisión de cada span** (`_emit_span`):

| Situación | Resultado |
|---|---|
| `end - start <= max_chunk_size` | 1 chunk `kind="python"` |
| No cabe y el nodo es `ClassDef` | regla P2 (split por miembros) |
| No cabe y es otra cosa (función gigante, imports...) | `sliding_window(...)` sobre el span |

**Regla P2 — split de una clase grande** (`_class_pieces`):

Divide la clase en piezas consecutivas que baldosan exactamente su span:

```
header   = [inicio_clase, inicio_primer_miembro)      # firma + docstring
member_i = [inicio_miembro_i, inicio_miembro_{i+1})   # un miembro cada uno
último   = [inicio_último_miembro, fin_clase)
```

- `inicio_miembro` = línea de su **primer decorador** si lo tiene (métodos
  decorados), si no su `lineno` → cortes siempre al inicio de línea.
- `_class_pieces` **verifica** que las piezas realmente cubren el span sin
  huecos; si algo no cuadra → devuelve `[]` → el llamador cae a
  `sliding_window` (nunca produce basura).
- Recursión acotada: una pieza que sea otra clase anidada grande vuelve a
  dividirse por sus miembros; cualquier otro miembro gigante → sliding.

Al final, `chunk_python` llama `validate_chunks(...)`: invariantes
garantizados por construcción **y** verificados antes de devolver.

### 3.2 ¿Qué es el fallback?

Es el **plan B cuando el AST no es utilizable** — el chunking no puede
fallar, solo puede degradar:

```python
try:
    tree = ast.parse(text)
except (SyntaxError, ValueError):
    tree = None
if tree is None or not tree.body:
    chunks = _fallback_chunks(...)   # sliding_window sobre TODO el fichero
```

Se activa en **tres casos**:

1. **`SyntaxError`** — el fichero Python está roto (`def broken(:`).
2. **`ValueError`** — `ast.parse` lo lanza si el fuente contiene bytes NUL.
3. **`tree.body` vacío** — el fichero parsea pero no hay sentencias de
   primer nivel (p. ej. un fichero con solo `# un comentario`).

Qué hace: `sliding_window(text, file_path, 0, len(text), max_chunk_size,
overlap)` → **todos los chunks `kind="sliding"`**, cobertura total del
fichero, todos ≤ max.

Filosofía: **nunca crashear, nunca dejar huecos, nunca superar 2000** — aunque
la segmentación quede peor (sin unidades lógicas). Estos ficheros son raros y
ya eran ingenuables de todos modos.

### 3.3 ¿Qué son las `sliding_window`?

Son el **último recurso de corte**: una ventana deslizante sobre
**caracteres**. Corta el rango `[start, end)` en trozos consecutivos de a lo
sumo `max_chunk_size` caracteres, donde cada ventana siguiente empieza
`overlap` caracteres **antes** de donde terminó la anterior.

Ejemplo con `len=12`, `max_chunk_size=5`, `overlap=2`:

```
posición:  0    5    8   11  12
ventana1:  [0....5)
ventana2:      [3....8)      ← empieza en 5-2 = 3
ventana3:          [6....11) ← empieza en 8-2 = 6
ventana4:             [9....12) ← empieza en 11-2 = 9
tamaños:   5    5    5    3   (todos ≤ 5, sin huecos)
```

Garantías por cada chunk producido:

- los 3 invariantes de la sección 0 (`0 <= start < end <= len(text)`,
  `end - start <= max_chunk_size`, `text[start:end] == chunk.text`),
- `kind="sliding"` (siempre),
- para consecutivos: `prev.end - cur.start == overlap`.

**¿Por qué por caracteres y no por líneas?** Decisión deliberada (sección
3.4 del plan): alinear por líneas + solape tiene casos límite de progreso (el
fichero tiene una línea de 1564 caracteres) y es fácil romper el ≤ 2000; la
ventana por caracteres lo garantiza **incondicionalmente**. Decisión que hay
que anotar en el README ("Design decisions").

**Dónde se usa (4 sitios):**

1. **Fallback** de `python_chunker` (fichero roto / solo comentarios).
2. **E1**: span de primer nivel que no cabe y no es una clase (función
   gigante...).
3. **P2**: miembro de clase que no cabe tras el split.
4. **Markdown `_flush_group`**: un párrafo aislado que supera
   `max_chunk_size`.

Nota de tuning (Fase 6): el `overlap` del CLI solo afecta a chunks *sliding*;
los grupos de párrafos markdown nunca se solapan entre sí.

---

## 4. Recall de la moulinette — ¿qué es exactamente?

**Recall@k** = de los sources "correctos" que declara el ground truth para una
pregunta, **qué proporción has traído dentro de tus primeros k resultados**.

Por pregunta:

```
recall@k = #sources correctos encontrados en tu top-k
           ────────────────────────────────────────────
           #sources correctos totales de la pregunta
```

(se promedia después sobre todas las preguntas del dataset; la moulinette lo
reporta para k = 1, 3, 5, 10).

### ¿Cuándo cuenta un source como "encontrado"?

Se comprueban DOS cosas:

1. **Mismo `file_path` — comparación verbatim.** Tu resultado debe decir
   exactamente `data/raw/vllm-0.10.1/docs/features/lora.md`. Un prefijo
   distinto (`vllm-0.10.1/...` sin `data/raw/`) **nunca casa** → recall 0 en
   esa pregunta.
2. **Solape de rangos con IoU ≥ 0.05**, donde
   `IoU = |A ∩ B| / |A ∪ B|` (A = rango ground truth, B = tu rango).

Ejemplo: ground truth `[1000, 2000]`, tu chunk `[1500, 3500]`:

```
intersección = 500,  unión = 2500  →  IoU = 0.2  ≥ 0.05  → ¡acierto!
```

La barra es **bajísima a propósito**: no hace falta clavar el span exacto,
solo "cubrir la región correcta del fichero correcto". Pero un chunk en otro
fichero o un IoU por debajo de 0.05 no suma nada.

### Umbrales obligatorios (los que hay que superar en Fase 6)

| Dataset | Umbral |
|---|---|
| **docs** | **≥ 80% recall@5** |
| **code** | **≥ 50% recall@5** |

### Comando oficial

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/<Scope>/<file>.json \
  data/datasets/AnsweredQuestions/<file>.json \
  --k 10 --max_context_length 2000
```

### ¿Por qué se comenta en la fase de chunking?

Porque **el chunk ES el source** que se evalúa. Un fallo de chunking se
traduce directamente en loss de recall:

- offsets/huecos malos → IoU 0 → nunca aciertas;
- chunk > 2000 caracteres → la moulinette invalida **toda la salida** de la
  corrida (incluso si el resto era perfecto);
- `file_path` con prefijo distinto → nunca casa.

Tu comando `evaluate` (Fase 5, `src/evaluation/recall.py` con `overlap_iou`)
reproduce este mismo cálculo **para tu iteración interna**; la métrica
oficial de la defensa es la de la moulinette, y tu código **nunca la importa
ni la llama**.

---

## 5. `markdown_chunker.py` — bloque Python dentro del markdown vs `kind="python"`

### Respuesta corta

**No distingue por lenguaje; distingue por estrategia. El campo `kind` dice
QUIÉN produjo el chunk, no de qué está hecho su contenido.**

| Fichero (extensión) | Chunker elegido | `kind` que emite |
|---|---|---|
| `.py` | `chunk_python` | `"python"` (o `"sliding"` si no cabe / fallback) |
| `.md` / `.txt` / `.rst` | `chunk_markdown` | `"markdown"` (o `"sliding"` si un párrafo no cabe) |

El dispatch ocurre en `chunk_file` **por extensión del fichero**, no por
detectar el idioma del contenido. Un snippet Python dentro de un `.md`:

````markdown
## Ejemplo de uso

```python
# esto es un comentario Python a columna 0
def hola() -> str:
    return "mundo"
```

Más texto después.
````

→ todo eso cae dentro de una **sección markdown** → chunks
`kind="markdown"`. El `# comentario` jamás produce un chunk
`kind="python"`.

### Entonces, ¿qué hace con los bloques de código Python del markdown?

Solo los usa para **no confundirlos con cabeceras**. El peligro concreto
(documentado en el plan): los `.md` de vLLM embeben snippets de Python cuyas
líneas `# comment` están a columna 0 y **encajarían con la regex de cabecera**
ATX (`_HEADER_RE = ^ {0,3}#{1,6}(?:\s|$)`), creando fronteras de sección
falsas (y partiría el código por la mitad).

La solución es una **máquina de estados de vallas (fences)** en
`_header_starts`, alimentada por `_fence_marker`:

1. Una línea cuyo texto (stripped) empieza por **≥ 3 `` ` `` o `~`**
   **abre** una valla → se recuerdan carácter y longitud
   (p. ej. ```` ``` ```` → `("`", 3)`).
2. **Mientras la valla está abierta**, cada línea se salta con `continue`:
   - no se compara contra `_HEADER_RE` → un `# comment` dentro de
     ```` ```python ```` **no crea frontera de sección**,
   - tampoco abre vallas anidadas (contador simple, no CommonMark completo).
3. Una línea que empieza por **el mismo carácter repetido ≥ la longitud de la
   valla** la **cierra** → se vuelve a detectar cabeceras normalmente.

Esto es exactamente lo que verifica el **test 35**: un bloque
```` ```python ```` con un `# comment` a columna 0 → el número de chunks
coincide solo con las cabeceras reales, cobertura completa.

Setext (`Title` / `=====`) y front matter YAML (`---`) **no** son especiales:
se tratan como texto plano (tests 39 y de front matter).

### Tabla final de `kind`

| `kind` | Quién lo emite | Cuándo |
|---|---|---|
| `"python"` | solo `chunk_python` | span de AST (P1/E1/P2) que cabe en `max_chunk_size` |
| `"markdown"` | solo `chunk_markdown` | sección por cabecera o grupo de párrafos (M1) que cabe |
| `"sliding"` | `sliding_window` (común a ambos) | cualquier cosa que **no cabe** (función gigante, párrafo gigante) o **fallback** (fichero `.py` roto / solo comentarios) |

Es decir: `"sliding"` **no** significa "es Python partido", significa "esto
lo partió la ventana deslizante porque no cabía de otra manera".

---

## 6. Autoevaluación — preguntas para defenderte

1. **¿Quién llama a `chunk_file` y por qué el indexer debe pasar rutas
   relativas?** — El indexer (Fase 3) al recorrer `data/raw/`; porque la
   moulinette compara `file_path` verbatim y un prefijo distinto jamás casa.
2. **Si un chunk mide 2001 caracteres, ¿qué pasa exactamente?** — La
   moulinette (con `--max_context_length 2000`) rechaza TODA la salida de la
   corrida, no solo ese source.
3. **¿Por qué `overlap >= max_chunk_size` es un error y no solo "raro"?** —
   Porque `cursor = chunk_end - overlap` no avanzaría → bucle infinito en
   `sliding_window`.
4. **¿Qué tres casos activan el fallback AST?** — `SyntaxError`,
   `ValueError` (bytes NUL) y `tree.body` vacío (fichero solo comentarios).
5. **Un `# comment` dentro de un ```` ```python ```` en un `.md`: ¿qué
   `kind` tiene y por qué no parte la sección?** — El trozo es de un chunk
   `kind="markdown"` (dispatch por extensión); la máquina de vallas lo salta
   para que no dispare `_HEADER_RE`.
6. **¿Por qué un comentario entre dos funciones se adhiere a la función
   siguiente?** — Regla P1: el span del nodo `n_i` empieza en
   `after(n_{i-1})`, o sea justo tras la línea final del nodo anterior. Todo
   lo que hay en ese hueco (comentarios intermedios, líneas en blanco) cae
   **dentro del span de `n_i`**, el nodo SIGUIENTE. Y es intencionado: un
   comentario justo encima de `def foo` suele documentar a `foo`.
7. **¿Qué mide el IoU ≥ 0.05 y por qué es tan bajo?** — El solape entre tu
   rango y el ground truth (`|A∩B|/|A∪B|`); bajo a propósito para no exigir
   clavar el span exacto, solo cubrir la región correcta del fichero
   correcto.
8. **¿Qué comprueba `validate_chunks` y quién lo llamará más adelante?** —
   Slice consistency, rango válido, tamaño ≤ max y orden creciente; lo
   llamarán el indexer (Fase 3) antes de persistir y el retriever (Fase 4)
   antes de devolver resultados.
9. **¿Por qué `Chunk` es un dataclass frozen y no pydantic?** — El subject
   solo exige pydantic en los modelos intercambiables entre etapas; el chunk
   es detalle interno, y frozen da inmutabilidad + comparabilidad `==` barata
   en tests.
10. **¿Qué gana la estrategia AST frente a un corte fijo por caracteres?** —
    Corta en unidades lógicas (función/clase/import con su docstring y
    comentarios), el vocabulario del índice queda coherente y el retrieval
    recupera "un tema entero"; el corte fijo solo queda como fallback.
