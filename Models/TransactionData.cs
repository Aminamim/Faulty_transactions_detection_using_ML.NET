using Microsoft.ML.Data;

namespace FraudDetectionAPI.Models
{
    public class TransactionData
    {
        [LoadColumn(0)] public string DATE { get; set; }
        [LoadColumn(1)] public string DESCRIPTION { get; set; }
        [LoadColumn(2)] public string ITEMNAME { get; set; }
        [LoadColumn(3)] public float QUANTITY { get; set; }
        [LoadColumn(4)] public float UNITCOST { get; set; }
        [LoadColumn(5)] public float STOCKONHAND { get; set; }
        [LoadColumn(6)] public float RUNNINGUNITCOST { get; set; }
        [LoadColumn(7), ColumnName("Label")] public bool ISFAULTY { get; set; }
    }
}
