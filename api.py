from janome.tokenizer import Tokenizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import joblib
from gensim.models import Word2Vec
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

model = Word2Vec.load('word2vec.gensim.model')
app = Flask(__name__)
CORS(app)

def japanese_tokenizer(text):
    t = Tokenizer()
    return [token.surface for token in t.tokenize(text)]

def search(query, top_n=30):
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    df = joblib.load('document_dataframe.pkl')
    
    query_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    return [(df.iloc[i]["DocumentID"], df.iloc[i]["Content"], cosine_similarities[i]) for i in related_docs_indices]

def get_top_similar_words(word, top_n=30):
    similar_words = model.wv.most_similar(word, topn=top_n)
    return similar_words

def is_single_word(text):
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(text))
    return_words = []

    if len(tokens) == 1 and tokens[0].part_of_speech.split(",")[0] == "名詞":
        similar_words = get_top_similar_words(tokens[0].base_form)
        for i in similar_words:
            search_words = search(i[0], top_n=20)
            for j in search_words:
                return_words.append((j[0], j[1], float(j[2]) * i[1]))  # タプル形式で統一

        return return_words
    return []

@app.route('/api/data', methods=['GET'])
def get_data():
    class_  = request.args.get("class_","guest")
    if "," in class_ : class_ = class_.split(",")
    else: class_ = [class_]
    soq     = request.args.get("soq","guest")
    if "," in soq: soq = soq.split(",")
    else: soq = [soq]
    date    = request.args.get("date","guest")
    if "," in date: date = date.split(",")
    else: date = [date]
    period  = request.args.get("period","1")
    if "," in period: period = [int(value) for value in period.split(",")]
    else: period = [int(period)]
    year    = request.args.get("year","1")
    print(year)
    if "," in year: year = [int(value) for value in year.split(",")]
    else: year = [int(year)]
    keyword = request.args.get("keyword","guest")

    json_data = {
        "class_"   : class_,
        "soq"     : soq,
        "date"    : date,
        "period"  : period,
        "year"    : year,
        "keyword" : keyword
    }

    search_results = search(keyword)
    search_resemble_results = is_single_word(keyword)
    searchs = sorted(search_results + search_resemble_results, key=lambda x: x[2], reverse=True)

    with open("search.json","r",encoding="utf-8") as file:
        data = json.load(file)

    results = []
    nums = []
    for i in searchs:
        if i[2] != 0.0:
            if (data[i[0] - 1]["class"] in class_) and (data[i[0] - 1]["semester_or_quarter"] in soq) and (data[i[0] - 1]["date"] in date) and (data[i[0] - 1]["period"] in period) and any(item in year for item in data[i[0] - 1]["year"]):
                ans = {}
                num = i[0] - 1
                if num not in nums:
                    nums.append(num)
                    ans["title"] = data[i[0] - 1]['title']
                    ans["link"]  = data[i[0] - 1]["syllabus_link"]
                    results.append(ans)

    if results is not None:
        json_data["results"] = results
    else:
        json_data["results"] = ["None"]

    print("return ")

    return jsonify(json_data)

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0")
