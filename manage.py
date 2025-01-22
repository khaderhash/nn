import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# تحميل البيانات
df = pd.read_csv('crime_data.csv')

# عرض الأعمدة للتحقق
print(df.columns)

# تنظيف النصوص في عمود 'Disposition'
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # إزالة HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # إزالة الأرقام والرموز
    text = text.lower()  # تحويل النص إلى حروف صغيرة
    return text

# تنظيف الأعمدة النصية في البيانات
df['Disposition'] = df['Disposition'].apply(clean_text)

# إزالة أي قيم مفقودة
df = df.dropna(subset=['Disposition', 'Category'])

# عرض بعض النصوص بعد التنظيف
print(df.head())

# تحويل الأعمدة الفئوية إلى قيم رقمية باستخدام LabelEncoder
label_encoder = LabelEncoder()

# تحويل الأعمدة الفئوية إلى قيم رقمية
df['Offender_Gender'] = label_encoder.fit_transform(df['Offender_Gender'])
df['Victim_Gender'] = label_encoder.fit_transform(df['Victim_Gender'])
df['Offender_Race'] = label_encoder.fit_transform(df['Offender_Race'])
df['Victim_Race'] = label_encoder.fit_transform(df['Victim_Race'])

# استخدام Tokenizer لتحويل النصوص إلى أرقام
tokenizer = Tokenizer(num_words=5000)  # تحديد 5000 كلمة الأكثر تكرارًا
tokenizer.fit_on_texts(df['Disposition'])

# تحويل النصوص إلى تسلسل رقمي
X = tokenizer.texts_to_sequences(df['Disposition'])

# تحديد الطول الأقصى للجملة
max_len = 150
X = pad_sequences(X, maxlen=max_len)

# تحويل الفئات (Category) إلى تمثيل رقمي
y = pd.get_dummies(df['Category']).values  # تحويل الفئات إلى تمثيل رقمي

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج
model = Sequential()

# طبقة Embedding لتحويل الكلمات إلى تمثيلات رقمية
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))

# إضافة طبقة LSTM
model.add(LSTM(100, return_sequences=True))

# إضافة طبقة CNN لاستخراج الأنماط
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))

# طبقة إضافية لـ LSTM
model.add(LSTM(100))

# طبقة Dropout لمنع الإفراط في التكيف
model.add(Dropout(0.5))

# طبقة Dense لتصنيف الفئات
model.add(Dense(y.shape[1], activation='softmax'))  # softmax لأن لدينا أكثر من فئة

# تجميع النموذج
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# عرض ملخص النموذج
model.summary()

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# تقييم النموذج على بيانات الاختبار
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# التنبؤ بالتصنيفات
y_pred = model.predict(X_test)

# عرض تقرير التصنيف
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes))

# عرض مصفوفة الارتباك
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

# رسم دقة التدريب مقابل التحقق
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# رسم الخسارة
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# حفظ النموذج المدرب
model.save('crime_analysis_model.h5')
