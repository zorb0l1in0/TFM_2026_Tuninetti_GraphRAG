import yaml
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_PATH = BASE_DIR / "../data/ner/ontologia/ontology.yaml"
OUTPUT_DIR = BASE_DIR / "sitio_ontologia"


def load_template(name):
    path = BASE_DIR / "templates" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<html><body><h1>{{NAME}}</h1>{{ENTITIES_LIST}}{{RELATIONS_LIST}}{{DOMAIN_BLOCK}}{{RANGE_BLOCK}}</body></html>"


def ontology_to_site(path_yaml, output_folder):
    # 1. Carga de datos
    yaml_file = Path(path_yaml)
    if not yaml_file.exists():
        print(f"❌ Error: archivo no encontrado en {yaml_file}")
        return

    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    entidades = data.get("entities", {})
    relaciones = data.get("relations", {})

    # 2. Preparación de carpetas
    out = Path(output_folder)
    ent_dir = out / "entidades"
    rel_dir = out / "relaciones"
    for d in [out, ent_dir, rel_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # ---------------- 3. GENERACIÓN DEL ÍNDICE GENERAL ----------------
    ent_links = "".join([f'<li><a href="entidades/{e}.html">{e}</a></li>' for e in entidades])
    rel_links = "".join([f'<li><a href="relaciones/{r}.html">{r}</a></li>' for r in relaciones])

    tmpl_index = load_template("index.html")
    index_html = (
        tmpl_index.replace("{{ENTITIES_LIST}}", ent_links)
        .replace("{{RELATIONS_LIST}}", rel_links)
    )

    (out / "index.html").write_text(index_html, encoding="utf-8")

    # ---------------- 4. LÓGICA DE RELACIONES (MAPEO) ----------------
    rels_by_ent = {e: {"domain": [], "range": []} for e in entidades}
    for rel, info in relaciones.items():
        for d in info.get("domain", []):
            if d in rels_by_ent:
                rels_by_ent[d]["domain"].append((rel, info.get("range", [])))
        for r in info.get("range", []):
            if r in rels_by_ent:
                rels_by_ent[r]["range"].append((rel, info.get("domain", [])))

    # ---------------- 5. PÁGINAS DE ENTIDADES ----------------
    tmpl_ent = load_template("entity.html")
    for name, info in entidades.items():
        # Relaciones de SALIDA
        dom_items = []
        for rel, targets in rels_by_ent[name]["domain"]:
            chips = "".join([f'<a class="chip" href="{t}.html">{t}</a>' for t in targets])
            dom_items.append(
                f'<div class="box">'
                f'<div class="box-header">'
                f'<span class="box-origin">{name}</span>'
                f'<span class="box-arrow">➔</span>'
                f'<a class="rel-pill" href="../relaciones/{rel}.html">{rel}</a>'
                f'</div>'
                f'<div class="box-targets">{chips}</div>'
                f'</div>'
            )
        domain_block = "".join(dom_items) or "<p class='empty'>Sin relaciones de salida.</p>"

        # Relaciones de ENTRADA
        ran_items = []
        for rel, sources in rels_by_ent[name]["range"]:
            chips = "".join([f'<a class="chip" href="{src}.html">{src}</a>' for src in sources])
            ran_items.append(
                f'<div class="box">'
                f'<div class="box-header">'
                f'<div class="box-targets" style="margin-bottom:0">{chips}</div>'
                f'<span class="box-arrow">➔</span>'
                f'<a class="rel-pill" href="../relaciones/{rel}.html">{rel}</a>'
                f'<span class="box-arrow">➔</span>'
                f'<span class="box-origin">{name}</span>'
                f'</div>'
                f'</div>'
            )
        range_block = "".join(ran_items) or "<p class='empty'>Sin relaciones de entrada.</p>"

        html = (tmpl_ent.replace("{{NAME}}", name)
                .replace("{{DESCRIPTION}}", info.get("description", "Sin descripción."))
                .replace("{{DOMAIN_BLOCK}}", domain_block)
                .replace("{{RANGE_BLOCK}}", range_block))

        (ent_dir / f"{name}.html").write_text(html, encoding="utf-8")

    # ---------------- 6. PÁGINAS DE RELACIONES ----------------
    tmpl_rel = load_template("relation.html")
    for rel, info in relaciones.items():
        dom_chips = "".join([f'<a class="chip chip-domain" href="../entidades/{d}.html">{d}</a>' for d in info.get("domain", [])])
        ran_chips = "".join([f'<a class="chip chip-range"  href="../entidades/{r}.html">{r}</a>' for r in info.get("range",  [])])

        dom_block = (
            f'<div class="chips-panel">'
            f'<div class="helper">Entidades origen</div>'
            f'<div class="chips-wrap">{dom_chips}</div>'
            f'</div>'
        ) if dom_chips else "<p class='empty'>Ninguno.</p>"

        ran_block = (
            f'<div class="chips-panel">'
            f'<div class="helper">Entidades destino</div>'
            f'<div class="chips-wrap">{ran_chips}</div>'
            f'</div>'
        ) if ran_chips else "<p class='empty'>Ninguno.</p>"

        html = (tmpl_rel.replace("{{NAME}}", rel)
                .replace("{{DESCRIPTION}}", info.get("description", "Sin descripción."))
                .replace("{{DOMAIN_BLOCK}}", dom_block)
                .replace("{{RANGE_BLOCK}}", ran_block))

        (rel_dir / f"{rel}.html").write_text(html, encoding="utf-8")

    print(f"✅ Sitio generado en: {out.absolute()}")


if __name__ == "__main__":
    ontology_to_site(ONTOLOGY_PATH, OUTPUT_DIR)