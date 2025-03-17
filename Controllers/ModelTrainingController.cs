using FraudDetectionAPI.Services;
using Microsoft.AspNetCore.Mvc;

namespace FraudDetectionAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ModelTrainingController : Controller
    {
        private readonly ILogger<ModelTrainingController> _logger;
        private ModelTrainingService _modelTrainingService;

        public ModelTrainingController(ILogger<ModelTrainingController> logger, ModelTrainingService modelTrainingService)
        {
            _logger = logger;
            _modelTrainingService = modelTrainingService;
        }

        [HttpGet]
        public IActionResult TrainModel()
        {
            _modelTrainingService.Train();
            return Ok();
        }
    }
}
