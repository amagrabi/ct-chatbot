"""

Run the chatbot locally in the console without the Will framework (WIP, just code samples so far).

"""

if __name__ == '__main__':

    import pandas as pd
    import pickle

    from get_data import get_data
    from predict.MarkovTextGenerator import MarkovTextGenerator
    from predict.ExpertPredictor import ExpertPredictor

    data_full = get_data()
    data_undersampled_recent = get_data(max_msgs_per_user=10000, undersampling_method='recent')
    data_undersampled_random = get_data(max_msgs_per_user=10000, undersampling_method='random')
    data_dummy = pd.DataFrame()

    # Training and saving MarkovTextGenerator data
    markov_text_generator = MarkovTextGenerator(data=data_full)
    markov_text_generator.save(save_dir='models')

    # Training and saving ExpertPredictor data
    expert_predictor = ExpertPredictor(data=data_undersampled_recent)
    expert_predictor.save(save_dir='models')

    # Loading data and using MarkovTextGenerator
    print('Generate some markov texts')
    markov_model = pickle.load(open('models/markov_model.pkl', 'rb'))
    markov_text_generator = MarkovTextGenerator(data=data_full, model=markov_model)
    print(markov_text_generator.generate_texts_str(n=3, replace_at=True))
    print('')

    # Loading data and using ExpertPredictor
    model_expert_predictor = pickle.load(open('models/model_expert_predictor.pkl', 'rb'))
    vectorizer_expert_predictor = pickle.load(open('models/vectorizer_expert_predictor.pkl', 'rb'))
    name2userid = pickle.load(open('models/name2userid.pkl', 'rb'))
    userid2name = pickle.load(open('models/userid2name.pkl', 'rb'))
    expert_predictor = ExpertPredictor(data=data_dummy,
                                       model=model_expert_predictor,
                                       vectorizer=vectorizer_expert_predictor,
                                       name2userid=name2userid,
                                       userid2name=userid2name)
    print('Generate experts')
    print(expert_predictor.predict_experts_str('ml', n=10, replace_at=True))
    print('')

    print('Generate Topics')
    print(expert_predictor.predict_topics_str('Amadeus Magrabi', n=5, replace_at=True))
    print('')
