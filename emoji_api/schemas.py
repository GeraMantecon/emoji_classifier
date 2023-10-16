from marshmallow import Schema, fields

class TweetSchema(Schema):
    text = fields.Str(required=True)

class PredictionSchema(Schema):
    emoji = fields.Str(dump_only=True)
    confidence = fields.Str(dump_only=True)