using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace mlnetWeatherForecasterWebAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //Configuration
            WebHost.CreateDefaultBuilder()
                .Configure(options => {
                    options.UseRouting();
                    options.UseEndpoints(routes => {
                        // Define prediction endpoint
                        routes.MapPost("/predict", PredictHandler);
                    });
                })
                .Build()
                .Run();
        }

        static async Task PredictHandler(HttpContext http)
        {
            var mlContext = new MLContext();

            //Define DataViewSchema for data preparation pipeline and trained model
            DataViewSchema modelSchema;

            // Load trained models
            ITransformer trainedModel = mlContext.Model.Load("minTempForecastModel.zip", out modelSchema);

            // Create time series engine
            var forecastEngine = trainedModel.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);

            // Deserialize HTTP request JSON body
            var body = http.Request.Body as Stream;
            var input = await JsonSerializer.DeserializeAsync<ModelInput>(body);

            // Predict with time series engine
            var prediction = forecastEngine.Predict(input, horizon: 7);

            // Return prediction as response
            var response = JsonSerializer.Serialize<ModelOutput>(prediction);
            http.Response.ContentType = "application/json";
            await http.Response.WriteAsync(response);
        }

        public class ModelInput
        {
            [LoadColumn(2)]
            public DateTime Date { get; set; }

            [LoadColumn(3)]
            public float MaxTemp { get; set; }

            [LoadColumn(4)]
            public float MinTemp { get; set; }

        }

        public class ModelOutput
        {
            public DateTime Date { get; set; }

            public float MaxTemp { get; set; }

            public float MinTemp { get; set; }

            public float[] ForecastTemp { get; set; }

            public float[] LowerBoundTemp { get; set; }

            public float[] UpperBoundTemp { get; set; }
        }
    }
}
