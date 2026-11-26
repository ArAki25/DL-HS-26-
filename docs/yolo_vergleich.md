# Vergleich: klassische Segmentierung vs. YOLO

Die Hauptpipeline findet Zellen ohne Objektdetektor (Otsu + Watershed, `src/segmentation.py`) und klassifiziert sie mit einem eigenen CNN. Dieser Vergleich prüft, ob ein gelernter Detektor besser wäre. Er **ersetzt nichts**: eigene Skripte, eigene Abhängigkeiten (`requirements-yolo.txt`), die bestehenden Skripte sind unverändert.

## Forschungsfragen

1. **Lokalisierung:** Findet YOLO die annotierten Zellen besser (Precision, Recall, vor allem Recall *infizierter* Zellen) als die Segmentierung, und bleibt das unter Domain Shift (Test = andere Kamera) so?
2. **Wo entsteht der Gewinn?** Ist ein Vorteil von YOLO in der Gesamtpipeline auf die besseren Boxen zurückzuführen oder darauf, dass YOLO auch klassifiziert?
3. **Kosten:** Laufzeit pro Bild, Modellgrösse, Annotationsbedarf (relevant für Smartphone-Mikroskope).

## Pipelines

| Name | Boxen | Klasse | beantwortet |
|---|---|---|---|
| `seg` | Otsu + Watershed | – | Baseline Lokalisierung |
| `yolo:<run>` | YOLO, Klassen ignoriert | – | F1 |
| `seg+cnn` | Otsu + Watershed | Zell-CNN | Hauptpipeline (= `predict.py`) |
| `yolo:<run>+cnn` | YOLO | dasselbe Zell-CNN | F2: nur der Detektor ist ausgetauscht |
| `yolo:<run> e2e` | YOLO (`--mode stage`) | YOLO selbst | F2: Single-Stage gegen Two-Stage |

`seg+cnn` gegen `yolo+cnn` ist die sauberste Ablation: der Klassifikator ist identisch, also kommt jede Differenz vom Detektor. `yolo+cnn` gegen `yolo e2e` zeigt, ob der YOLO-Klassifikationskopf mit dem spezialisierten Zell-CNN mithalten kann.

## Was den Vergleich fair macht

| Faktor | Umsetzung |
|---|---|
| Daten-Split | `assign_splits` (Seed 42), identisch zur Hauptpipeline; YOLO sieht val/test nie beim Training |
| Farbnormalisierung | derselbe `StainNormalizer` (aus `normalization.json` bzw. dem Zell-Checkpoint), für Export *und* Inferenz; Abweichungen werden gewarnt |
| Vergrösserung | Bilder werden vor YOLO auf feste Zellgrösse skaliert (`estimate_cell_diameter`, ohne Annotation), dieselbe Idee wie bei der Segmentierung. Ablation: `prepare_yolo.py --cell-px 0` |
| Zuordnung | ein Code für alle (`src/detection_eval.py`): greedy nach IoU, identisch zu `evaluate_segmentation.py` |
| `difficult`-Zellen | weder GT noch Hintergrund: Treffer darauf werden für *alle* Methoden ignoriert (wie COCO "crowd") |
| Schwelle | YOLO-Konfidenz wird auf **val** gewählt (max. Lokalisierungs-F1) und für **test** übernommen. AP50 wird zusätzlich schwellenunabhängig berichtet |
| Unsicherheit | paired Bootstrap über Bilder (2000 Resamples): CI für jede Metrik und für die Differenz zur Hauptpipeline. Bilder sind die unabhängige Einheit, nicht Zellen |
| Vortraining | Standard: COCO-vortrainiert. `--scratch` als fairere Variante gegenüber dem von Grund auf trainierten `simplecnn` |

**Was sich nicht angleichen lässt:** YOLO lernt aus den Boxen (überwacht), die Segmentierung braucht keine Annotationen, nur drei einmal kalibrierte Parameter. Ein YOLO-Gewinn kostet also Annotationsaufwand, und im Bericht gehört das als Trade-off dazu, nicht als Fussnote. Ausserdem haben Segmentierungs-Boxen einen systematischen Grössenfehler (die Maske verpasst den blassen Zellrand). Deshalb `--iou 0.3` und `--iou 0.75` als Sensitivitätsanalyse mitlaufen lassen: Wenn sich die Rangfolge nur bei 0.75 dreht, ist das ein Box-Passform-Effekt und keiner der Erkennung.

## Metriken

- **Lokalisierung:** Precision, Recall, F1, Recall infizierter Zellen (*was hier fehlt, kann kein Klassifikator retten*), relativer Zählfehler pro Bild, AP50 (nur YOLO, die Segmentierung hat keinen Score).
- **Gesamtpipeline:** Detektions-Konfusionsmatrix mit Hintergrund-Zeile/-Spalte. Ein Treffer zählt nur, wenn Box (IoU ≥ 0.5) *und* Klasse stimmen. Daraus: F1 pro Stadium, `macro_f1_present` (wie in `engine.py`), F1/Recall infiziert, **Parasitämie-MAE** in Prozentpunkten (klinisch die eigentliche Zielgrösse) und Sensitivität auf Bild-Ebene.
- **Kosten:** ms/Bild für Detektor bzw. Segmentierung, ms/Zelle für das CNN, auf demselben Gerät.

## Hypothesen (vor dem Experiment festhalten, nicht gemessen)

1. **Lokalisierung, val:** YOLO hat höheren Recall, vor allem bei Zellhaufen, wo die Watershed-Trennung über die Distanztransformation an Grenzen kommt (sich überlappende Zellen haben keinen sauberen Sattel). Bei isolierten Zellen ist die Segmentierung fast gleich gut.
2. **Lokalisierung, test:** Die Lücke kann in beide Richtungen gehen. Otsu passt seine Schwelle pro Bild an und ist damit robust gegen Helligkeitsänderungen. YOLO hat die Test-Kamera nie gesehen und hängt stärker an Normalisierung und Augmentation. Ein YOLO, das auf val klar gewinnt und auf test einbricht, wäre ein wichtiges Ergebnis für den Bericht.
3. **Gesamtpipeline:** `yolo+cnn` ≥ `seg+cnn` (bessere Boxen, gleicher Klassifikator), aber `yolo e2e` < `yolo+cnn` bei seltenen Stadien (Schizont, Gametozyt). Gründe: Der YOLO-Kopf hat kein Class-Balancing (kein Sampler, kein Class-Weight; die Klassifikations-BCE wird von rbc dominiert), er sieht jede Zelle mit ca. 48 px statt 96 px, und Mosaic/Skalierung verkleinern sie weiter. Zwei-Stufen-Pipelines sind bei extremem Ungleichgewicht häufig im Vorteil.
4. **Kosten:** Die Segmentierung ist auf der CPU schnell genug und braucht keine Gewichte. YOLO-n ist mit ca. 2-3 M Parametern trotzdem smartphone-tauglich. Entscheidend ist, ob der Recall-Gewinn den Annotations- und Lizenzaufwand (AGPL) rechtfertigt.

## Ablauf für den Bericht

```bash
pip install -r requirements-yolo.txt
python scripts/prepare_yolo.py --mode cell && python scripts/prepare_yolo.py --mode stage
python scripts/train_yolo.py --mode cell                    # vortrainiert
python scripts/train_yolo.py --mode cell --scratch          # Ablation Vortraining
python scripts/train_yolo.py --mode stage --epochs 150
python scripts/compare_detection.py --split val --show 3 \
    --yolo runs/yolo/<cell>/weights/best.pt --yolo runs/yolo/<cell_scratch>/weights/best.pt \
    --yolo runs/yolo/<stage>/weights/best.pt --cell-model runs/<stage-cnn>/best.pt
python scripts/compare_detection.py --split val --iou 0.3 --out runs/compare/val_iou03 ...
python scripts/compare_detection.py --split test ...        # einmal, ganz am Schluss
```

Für `--split test` werden die Schwellen aus `runs/compare/val/operating_points.json` gelesen. Gibt es dort für einen Run keine Schwelle, bricht das Skript ab, statt auf test zu optimieren.

## Grenzen

- Val hat nur ca. 180 Bilder, test 120: Die Bootstrap-CIs für seltene Stadien (wenige Dutzend Zellen) sind breit. Differenzen, deren CI die 0 enthält, nicht als Gewinn verkaufen.
- Es gibt nur einen Seed pro Modell. Für belastbare Aussagen 3 Seeds pro YOLO-Konfiguration trainieren und die Streuung berichten.
- Der YOLO-Recall hängt von `max_det` ab (hier 1000, Ultralytics-Standard wäre 300) und bei NMS-basierten Modellen (YOLO11) von der NMS-IoU. YOLO26 arbeitet standardmässig ohne NMS.
