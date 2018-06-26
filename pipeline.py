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
    data_resampled = get_data(max_msgs_per_user=5000, undersampling_method='recent',
                                             boost_users_in_range=(1, 1500), boost_factor=2)
    data_dummy = pd.DataFrame()

    # Training and saving MarkovTextGenerator data
    markov_text_generator = MarkovTextGenerator(data=data_full)
    markov_text_generator.save(save_dir='models')

    # Training and saving ExpertPredictor data
    expert_predictor = ExpertPredictor(data=data_resampled, min_df=5, max_df=0.05)
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

    # Test some example experts
    threshold = 0.01
    n = 5
    test_words = ['ops', 'scala', 'ml', 'machine learning', 'smoking', 'party', 'bike', 'running', 'beer',
                  'drunk', 'blog posts', 'jvm', 'graphql', 'Döner', 'cooking', 'ElasticSearch', 'Support',
                  'merchant center', 'kicker', 'ping pong', 'German', 'food', 'Russian', 'American',
                  'prosecco']
    for test_word in test_words:
        print(f'Generate experts for {test_word}:')
        print(expert_predictor.predict_experts_str(test_word, n=n, replace_at=True, threshold=threshold))
        print('')

    # Test some example topics
    n = 5
    users = ['Amadeus Magrabi', 'Hajo Eichler', 'Konrad Fischer']
    for user in users:
        print(f'Generate topics for {user}:')
        print(expert_predictor.predict_topics_str(user, n=n, replace_at=True))
        print('')

    # Upload trained models to bucket
    # gsutil -m cp -r models gs://ctp-playground-ml-datasets/hipchat
