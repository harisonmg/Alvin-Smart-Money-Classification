# Best result so far
## Model
CatBoost with default parameters

## Preprocessing
The default preprocessing pipeline in version `0.1.0`.

### Actions
- Count vectorize merchant name
- Obtain day, day of week, month and hour of day from purchase timestamp
- Log transform purchase amount and user income
- Ordinal encode user ID and gender
- Impute missing age with a constant
- Select household size and `IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY`

## Score
- Metric: Log loss
- Cross validation score: 1.54217 (std: NaN)
- Public leaderboard score: 1.468897 (rank: 50/221)
- Private leaderboard score: 1.584807 (rank: 48/221)
