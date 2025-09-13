import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pii_masker import process_jsonl_first_1000
    process_jsonl_first_1000("enrichedwstreet_with_pii_dois.jsonl", "masked_first_1000f.jsonl")
    return


if __name__ == "__main__":
    app.run()
