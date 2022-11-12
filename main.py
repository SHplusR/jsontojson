import json
import re
from konlpy.tag import Okt
import pickle
import numpy as np

file_path = "./maioutput.json"
with open(file_path, 'r', encoding='UTF-8') as file:

    data = json.load(file)

okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다','로','것','고','원','보','젠']
max_len = 50

with open('./tokenizer.pickle','rb') as handle:
  tokenizer = pickle.load(handle)

def review_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True)
  new_sentence = [word for word in new_sentence if not word in stopwords]
  encoded = tokenizer.texts_to_sequences([new_sentence])
  return encoded

# data[i]마다 review만 있는 dict 따로 만들고, 토큰화 한 애들만 있는 dict를 따로 만들고 이를 조합한 애를 만든다.
# data[i]마다 반복해야할 과정 (1) review를 가져와 list로 만든다. (2) data[i]의 placereview list 를 clear한다. (3) 1의 리뷰를 review_predict에 넣어 토큰화를 한 list를 만든다.
# (4) 1과 3을 list로 만든다. ex) secont_dict (5) review_dict와 secont_dict를 zip화 하여 새로운 dict를 만든다. (6) 이를 placereview에 append한다.

data_len = len(data)
# print(data_len)
review_dict = ["review","token"]

# reviewlist = ['a','b','c']
# tokenlist = [[12,3],[4,5],[6,7]]
# new = list(zip(reviewlist,tokenlist))
# print(new)
# icecream = dict(zip(review_dict, new))
# print(icecream)

# secont_dict = ["사장님 너무 친절하시고 음식도 맛있어요. 가게가 조용해서 힐링하다가갑니다. 단골예약이에요~",[86, 32, 9, 8, 147, 71, 776, 377, 1824, 73, 1461, 213, 408]]
# icecream = dict(zip(review_dict, secont_dict))
# print(icecream)

# print(data[0]['placeReviews'][0])
# print(review_predict(data[0]['placeReviews'][0]))
#
for i in range(0,data_len):
    review_len = len(data[i]['placeReviews'])
    # data[i]['placeReviews'].clear()
    for j in range(0,review_len):
        review_list = []
        review=data[i]['placeReviews'][j]
        review_list.append(review)

        review_token = review_predict(review)
        review_list.append(np.concatenate(review_token).tolist())

        new_list = dict(zip(review_dict,review_list))
        data[i]['placeReviews'].append(new_list)

print(data[0]['placeReviews'])


with open('./tokenoutput.json', 'w', encoding='utf-8') as make_file:

    json.dump(data, make_file)
