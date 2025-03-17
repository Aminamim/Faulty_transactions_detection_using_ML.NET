namespace FraudDetectionAPI.Models
{
    public class TransactionDataTransformed
    {
        public float[] DATE { get; set; }  // Encoded as numerical vectors
        public float[] DESCRIPTION { get; set; }  // Encoded as numerical vectors
        public float[] ITEMNAME { get; set; }  // Encoded as numerical vectors
        public float QUANTITY { get; set; }
        public float UNITCOST { get; set; }
        public float STOCKONHAND { get; set; }
        public float RUNNINGUNITCOST { get; set; }
    }
}
