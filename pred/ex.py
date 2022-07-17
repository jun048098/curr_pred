import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from konlpy.tag import Mecab
tokenizer = Mecab('C:\mecab\mecab-ko-dic')


# 데이터 전처리

# 데이터 경로 정의
curr = 'C:/Users/jun04/PycharmProjects/curr_pred7/pred/curr_list.csv'
data = pd.read_csv(curr)

# X데이터 정제
# 각각 토큰화한 리스트
def make_x(x_data):
    return tokenizer.morphs(x_data)
X_ = data.iloc[:,2].apply(make_x)

# 모든 토큰들 (중복)
X_word = []
for i in X_:
  X_word.extend(i)
X_word = { v:i for i,v in enumerate(np.unique(X_word))}  # X 사전

def make_label(x):
  x_list = []
  for i in x:
    x_list.append(X_word[i])
  return x_list
X = X_.apply(make_label)


from keras.preprocessing.sequence import pad_sequences
maxlen = X.apply(len).max()
X = pad_sequences(X,maxlen=maxlen)

# y 데이터
y_ = data.학과

# y 사전
y_word = { v:i for i,v in enumerate(np.unique(y_))}

y_dic = {v: k for k, v in y_word.items()}

def make_label_y(x):
  y_list = []
  for i in x:
    y_list.append(y_word[i])
  return y_list

Y = make_label_y(y_)

# 원핫인코딩 y데이터
Y = to_categorical(Y)

from sklearn.model_selection import train_test_split

# X,Y 학습데이터 완성
X_train, X_test, y_train, y_test = train_test_split(X, Y )

# 모델 구축 및 학습
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

# 인공 신경망 구성
model = Sequential()
model.add( Embedding(2234, 8, input_length=13 ) )
model.add( Flatten() )
model.add( Dense(135, activation='softmax') )

# 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping # 조기 학습 종료
early_stopping_callback =EarlyStopping(patience=5)

model.fit(X_train,
          y_train,
          batch_size =20,
          epochs=200,
          validation_data=(X_test, y_test),
          callbacks = [early_stopping_callback] )

# 예측 ->
model.evaluate(X_test, y_test)

# 모델 활용
def curr_predict(x,most_similary=1):
  tmp = []
  for i in tokenizer.morphs(x):
    if X_word.get(i) is None:
      tmp.append(0)
    else:
      tmp.append(X_word.get(i))
  tmp =  pad_sequences(np.array(tmp).reshape(1,-1),maxlen=maxlen)
  if most_similary == 1:
    pred = np.argmax(model.predict(tmp))
    pred = y_dic[pred]
    return pred
  else:
    pred = np.argsort(model.predict(tmp))
    pred = pred.reshape(-1,)
    pred = pred[::-1][:most_similary]
    classes = []
    for i in range(most_similary):
      pre = y_dic[pred[i]]
      classes.append(pre)
    return classes
print(curr_predict('건축학사'))