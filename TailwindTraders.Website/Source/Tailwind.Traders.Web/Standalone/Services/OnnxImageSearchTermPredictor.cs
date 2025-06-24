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

        public Task<string> PredictSearchTerm(Stream imageStream)
        {
            DenseTensor<float> data = ConvertImageToTensor(imageStream);
            var input = NamedOnnxValue.CreateFromTensor<float>("data", data);
            using var output = session.Run(new[] { input });
            var prediction = output.First(i => i.Name == "classLabel").AsEnumerable<string>().First();
            return Task.FromResult(prediction);
        }

        private DenseTensor<float> ConvertImageToTensor(Stream imageStream)
        {
            var data = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

            using (var image = Image.Load<Rgba32>(imageStream))
            {
                image.Mutate(ctx => ctx.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Stretch
                }));

                for (int y = 0; y < image.Height; y++)
                {
                    var pixelRowSpan = image.GetPixelRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        var color = pixelRowSpan[x];
                        // ONNX model often expects channels in order: R, G, B (adjust if your model expects BGR)
                        data[0, 0, y, x] = color.R / 255f; // Normalize pixel to [0,1]
                        data[0, 1, y, x] = color.G / 255f;
                        data[0, 2, y, x] = color.B / 255f;
                    }
                }
            }

            return data;
        }
    }
}
