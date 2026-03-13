from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

TARGET_COLUMNS = {
    "name": "student name full name learner candidate",
    "attendance": "student attendance percentage presence attendance rate",
    "marks": "student marks exam score grade result",
    "study_hours": "study hours study time learning hours preparation time"
}

def clean_column(col):
    col = col.lower()
    col = col.replace("_", " ")
    col = col.replace("%", " percentage")
    col = re.sub(r"[^a-zA-Z0-9 ]", "", col)
    return col.strip()

def map_columns(df):

    mapped = {}

    for target, description in TARGET_COLUMNS.items():

        best_score = 0
        best_column = None

        for col in df.columns:

            cleaned = clean_column(col)

            score = util.cos_sim(
                model.encode(cleaned, convert_to_tensor=True),
                model.encode(description, convert_to_tensor=True)
            )

            score = score.item()

            if score > best_score:
                best_score = score
                best_column = col

        mapped[target] = best_column

    return mapped
