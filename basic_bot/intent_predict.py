try:       
        import warnings
        warnings.filterwarnings("ignore")
        
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        from keras.models import load_model
        from preprocessor import nlp
        
        from preprocessor import pad_vec_sequences, labels
        import spacy
        import numpy as np
        import sys
        from dependency_tree import to_nltk_tree , to_spacy_desc

except Exception as inst:
        a = 6
        print('Exception occured')
        print(inst)




def process_labels(label):
	if(label == 'greetings'):
		return greetings_ans
	if(label == 'thanks'):
		return thanks_ans
	if(label == 'about_dl'):
	        return about_dl_ans
	if(label == 'about_event'):
	        return about_events_ans
	if(label == 'about_office_time'):
	        return about_office_timimgs_ans
	if(label == 'about_weather_ans'):
	        return about_office_timimgs_ans
	return 'Could not detect intent'


#print(labels)

#-----------------------------------------------------------
#responses to the intents

greetings_ans = 'Hello'

thanks_ans = 'Welcome'


about_dl_ans =  'Deep Learning is a class of machine learning algorithms that use multiple layers to extract features. Higher level features are extracted from lower level features to understand the input. Applications of Deep Learning include - Automatic Speech Recognition, Image Processing, Natural Language Processing and many more.'

about_events_ans = 'We keep on organizing Faculty Development Programs and Workshops in colleges. We also organize internal knowledge sharing sessions.'


about_office_timimgs_ans = 'We work from 10 am to 6 pm'

about_weather_ans = "I'll have to check with the weather man"

#------------------------------------------------------------

nb_classes = len(labels)

#load the model to be tested.
model = load_model('backup/intent_models/model2.h5')


def predict(text):
  test_vec_seq = [] #list of all vectorized test queries
  test_ent_seq = [] #list of lists of entities in each test query
  test_seq = [] #list of all test queries

  test_text = text
  test_seq.append(test_text)
  #vectorize text.
  test_doc = nlp(test_text)
  test_vec = []
  for word in test_doc:
    test_vec.append(word.vector)
  test_vec_seq.append(test_vec)
  test_ent_seq.append(test_doc.ents)
    
  #convert all the sentences into matrices of equal size.
  test_vec_seq = pad_vec_sequences(test_vec_seq)
  #get predictions
  prediction = model.predict(test_vec_seq)

  label_predictions = np.zeros(prediction.shape)


  m = max(prediction[0])
  print(m)
  p = np.where(prediction[0] > 0.55 * m)	# p collects possible sub intents
  q = np.where(prediction[0] == m)	#q collects intent
  label_predictions[0][p] = 1
  label_predictions[0][q] = 2


  for x in range(len(label_predictions[0])):
    if label_predictions[0][x] == 2 :
      print(process_labels(labels[x]))
      #print(" Detected intent: ",labels[x])





while True:
  text = input("Yes?\n")
  predict(text)

	
