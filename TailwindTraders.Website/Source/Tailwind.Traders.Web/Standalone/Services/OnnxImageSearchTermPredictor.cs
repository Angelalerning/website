using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace Tailwind.Traders.Web.Standalone.Services
{
    public class OnnxImageSearchTermPredictor : IImageSearchTermPredictor
    {
        private readonly ILogger<OnnxImageSearchTermPredictor> logger;
        private readonly InferenceSession session;

        public OnnxImageSearchTermPredictor(IWebHostEnvironment environment, ILogger<OnnxImageSearchTermPredictor> logger)
        {
            this.logger = logger;
            var filePath = Path.Combine(environment.ContentRootPath, "Standalone/OnnxModels/products.onnx");
            session = new InferenceSession(filePath);
        }

        public Task<st
