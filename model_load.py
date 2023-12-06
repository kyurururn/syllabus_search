from janome.tokenizer import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def japanese_tokenizer(text):
    t = Tokenizer()
    return [token.surface for token in t.tokenize(text)]

# Excelファイルの読み込み（適宜ファイルパスとシート名を指定）
file_path = 'Book1.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# TF-IDF Vectorizerの初期化
vectorizer = TfidfVectorizer(tokenizer=japanese_tokenizer)  # verboseを1に設定

# ドキュメントのTF-IDF行列を生成
tfidf_matrix = vectorizer.fit_transform(df['Content'])

# モデルを保存
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(df, 'document_dataframe.pkl')