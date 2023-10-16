from flask_smorest import Blueprint, abort
from schemas import TweetSchema, PredictionSchema
from flask.views import MethodView
from resources import tweet_classifier

blp = Blueprint("tweets",__name__,description="Operations on tweets.")
classifier = tweet_classifier.TweetClassifier()

@blp.route("/tweets")
class Tweets(MethodView):
    
    @blp.arguments(TweetSchema)
    @blp.response(200,PredictionSchema)
    def get(self,tweet_data):
        try:
            clean_tweet = classifier.clean_tweet(tweet_data['text'])
            prediction = classifier.predict(clean_tweet)
            prediction = PredictionSchema().from_dict(prediction)
            return prediction
        except Exception as ex:
            abort(417, message='Error processing request.')