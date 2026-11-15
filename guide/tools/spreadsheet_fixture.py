"""Invented workbooks matching the research notebook's fixed experimental design.

These are test data, not a reconstruction or validation of the private experiment.
"""
from pathlib import Path
import sys
import pandas as pd


def create(destination):
    destination = Path(destination)
    memory = destination / "Memory experiment"
    final = memory / "Data Analysis/Final data files/Completed_Recall_Data2024.xlsx"
    scoring = memory / "SCORING_MEMORY"
    ratings = destination / "Similarity Exp 1 - UK Sample/Juliette's stuff"
    for p in [final.parent, scoring, ratings]:
        p.mkdir(parents=True, exist_ok=True)
    negative = list(range(1, 25))
    neutral = [*range(25, 48), 63]
    labels = {**{v: f"neg{i}" for i,v in enumerate(negative, 1)}, **{v: f"neu{i}" for i,v in enumerate(neutral, 1)}}
    rows = {name: [] for name in ["pres_itemnos", "pres_itemnos_valence", "pres_catlabels", "recalls", "rec_itemnos", "rec_itemnos_valence", "rec_catlabels"]}
    for subject in range(1, 124):
        group = subject % 2
        for number in [1+2*group, 2+2*group, 5+2*group, 6+2*group]:
            if number <= 4:
                start = (number-1)*6
                items = negative[start:start+6] + neutral[start:start+6]
            else:
                items = (negative if number in [5,7] else neutral)[group*12:group*12+12]
            key = {"participant_number": subject, "List_Number": number}
            for name in rows:
                row = dict(key)
                if name.startswith("pres"):
                    values = items if name == "pres_itemnos" else [labels[v] if name.endswith("valence") else (101 if v in negative else 100) for v in items]
                    row.update({f"Pres{i}": v for i,v in enumerate(values, 1)})
                else:
                    for i in range(1, 15):
                        v = items[i-1] if i <= 4 else None
                        row[f"OP{i}"] = (i if name == "recalls" else v if name == "rec_itemnos" else labels[v] if name.endswith("valence") else 101 if v in negative else 100) if v else None
                    row["extra"] = None
                rows[name].append(row)
    with pd.ExcelWriter(final) as writer:
        for name, records in rows.items():
            pd.DataFrame(records).to_excel(writer, sheet_name=name, index=False)
    for name in ["rec_itemnos_juliette.xlsx", "rec_itemnos_dusan.xlsx"]:
        pd.DataFrame(rows["rec_itemnos"]).to_excel(scoring / name, index=False)
    empty = pd.DataFrame(columns=["participant_number", "List_Number", *[f"OP{i}" for i in range(1,15)], "extra"])
    empty.to_excel(scoring / "scoring_discrepancies.xlsx", sheet_name="Sheet 1", index=False)
    empty.to_excel(scoring / "scoring_discrepancies_filtered_FINAL.xlsx", sheet_name="actual disagreements", index=False)
    for kind in ["valence", "arousal"]:
        record = {"row": 1, "PrivateID": "synthetic"}
        for i, v in enumerate([*negative, *neutral]):
            filename = ("NegImage" if v in negative else "NeuImage") + labels[v][3:] + ".JPG"
            record[filename] = 1 + (i % 9) * 0.7 + (0.2 if kind == "arousal" else 0)
        pd.DataFrame([record, dict(record, row=2, PrivateID="synthetic-2")]).to_csv(ratings / f"{kind}.csv", index=False)


if __name__ == "__main__":
    create(sys.argv[1])
