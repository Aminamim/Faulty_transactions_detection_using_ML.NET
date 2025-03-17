using Microsoft.ML.Data;

namespace FraudDetectionAPI.Models
{
    public class TransactionPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsFaulty { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }
}
