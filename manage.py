import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %% 

# قراءة البيانات
data = pd.read_csv("crime_data.csv")
print(data.head())  # لمعرفة شكل البيانات

# التحقق من الأعمدة المطلوبة
if 'Victim_Fatal_Status' not in data.columns:
    raise ValueError("البيانات يجب أن تحتوي على العمود 'Victim_Fatal_Status'.")

# %% 
# إعداد البيانات وتنظيف النصوص
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # تحويل النص إلى حروف صغيرة
    text = ' '.join([word for word in text.split() if word not in stop_words])  # إزالة الكلمات غير المهمة
    return text

data['cleaned_text'] = data['Victim_Fatal_Status'].apply(clean_text)

# %% 

# تحويل النصوص إلى قيم رقمية باستخدام Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['cleaned_text'])
sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
word_index = tokenizer.word_index
print(f"عدد الكلمات الفريدة: {len(word_index)}")

# تحديد الحد الأقصى لطول النصوص
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)
y = data['Victim_Fatal_Status']  # تأكد من أن هذا العمود يحتوي على القيم الصحيحة

# %% 
# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيق SMOTE على البيانات (لتوازن الفئات)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# %% 

# إنشاء نموذج الشبكة العصبية
nn_model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_length),  # طبقة التضمين
    LSTM(64, return_sequences=True),  # طبقة LSTM
    Dropout(0.2),  # طبقة لتجنب الإفراط في التعلم
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # إخراج ثنائي
])

# تجميع النموذج
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
nn_model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, validation_split=0.2)

# %% 
# تقييم النموذج
nn_y_pred = nn_model.predict(X_test)
nn_y_pred = (nn_y_pred > 0.5)  # إذا كان التصنيف ثنائي
print("NN Accuracy:", accuracy_score(y_test, nn_y_pred))
print("Precision:", precision_score(y_test, nn_y_pred, average='weighted'))
print("Recall:", recall_score(y_test, nn_y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, nn_y_pred, average='weighted'))

# %% 
# رسم مصفوفة الالتباس
cm = confusion_matrix(y_test, nn_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Neural Network')
plt.show()
