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
                                             boost_users_in_range=(1, 1500), boost_factor=1.5)
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

    # Test some examples
    print('Generate experts for Ops')
    print(expert_predictor.predict_experts_str('Ops', n=10, replace_at=True))
    print('')
    print('Generate experts for Scala')
    print(expert_predictor.predict_experts_str('scala', n=10, replace_at=True))
    print('')
    print('Generate experts for ml')
    print(expert_predictor.predict_experts_str('ml', n=10, replace_at=True))
    print('')
    print('Generate experts for Machine Learning')
    print(expert_predictor.predict_experts_str('Machine Learning', n=10, replace_at=True))
    print('')
    print('Generate experts for smoking')
    print(expert_predictor.predict_experts_str('smoking', n=10, replace_at=True))
    print('')
    print('Generate experts for party')
    print(expert_predictor.predict_experts_str('party', n=10, replace_at=True))
    print('')
    print('Generate experts for beer')
    print(expert_predictor.predict_experts_str('party', n=10, replace_at=True))
    print('')
    print('Generate experts for drunk')
    print(expert_predictor.predict_experts_str('drunk', n=10, replace_at=True))
    print('')
    print('Generate experts for blog posts')
    print(expert_predictor.predict_experts_str('blog posts', n=10, replace_at=True))
    print('')
    print('Generate experts for jvm')
    print(expert_predictor.predict_experts_str('jvm', n=10, replace_at=True))
    print('')
    print('Generate experts for graphql')
    print(expert_predictor.predict_experts_str('graphql', n=10, replace_at=True))
    print('')
    print('Generate experts for Döner')
    print(expert_predictor.predict_experts_str('Döner', n=10, replace_at=True))
    print('')
    print('Generate experts for cooking')
    print(expert_predictor.predict_experts_str('cooking', n=10, replace_at=True))
    print('')
    print('Generate experts for elasticsearch')
    print(expert_predictor.predict_experts_str('elasticsearch', n=10, replace_at=True))
    print('')
    print('Generate experts for support')
    print(expert_predictor.predict_experts_str('support', n=10, replace_at=True))
    print('')
    print('Generate experts for merchant center')
    print(expert_predictor.predict_experts_str('merchant center', n=10, replace_at=True))
    print('')
    print('Generate experts for kicker')
    print(expert_predictor.predict_experts_str('kicker', n=10, replace_at=True))
    print('')
    print('Generate experts for table tennis')
    print(expert_predictor.predict_experts_str('table tennis', n=10, replace_at=True))
    print('')


    print('Generate Topics Amadeus')
    print(expert_predictor.predict_topics_str('Amadeus Magrabi', n=5, replace_at=True))
    print('')
    print('Generate Topics Hajo')
    print(expert_predictor.predict_topics_str('Hajo Eichler', n=5, replace_at=True))
    print('')
    print('Generate Topics Simon')
    print(expert_predictor.predict_topics_str('Simon White', n=5, replace_at=True))
    print('')

    # Upload trained models to bucket
    # gsutil -m cp -r models gs://ctp-playground-ml-datasets/hipchat
