from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
import uvicorn
from fastapi.responses import HTMLResponse

# Import the ML model function
# Assuming automated_regression_analysis_best_model is in model.py
from model import automated_regression_analysis_best_model

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Upload</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f4f4f9;
            }
            .container {
                width: 100%;
                max-width: 500px;
                background: #fff;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 10px;
            }
            h1 {
                font-size: 24px;
                text-align: center;
                margin-bottom: 20px;
            }
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload CSV File</h1>
            <form action="/upload_csv/" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept=".csv" required>
                <input type="submit" value="Upload and Process">
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Read the file into a string
    content = await file.read()

    # Convert the string into a pandas dataframe
    csv_data = StringIO(content.decode('utf-8'))
    df = pd.read_csv(csv_data)

    # Save the DataFrame as a CSV file locally to pass to the model (optional)
    temp_file_path = "temp_data.csv"
    df.to_csv(temp_file_path, index=False)

    # Run the model on the uploaded CSV file
    best_model_pipeline, processed_df = automated_regression_analysis_best_model(temp_file_path, test_size=0.2, k_best_features=5)

    # Return the best model result and print it on the HTML page
    result = f"Best Model: {best_model_pipeline}"

    # Generate an HTML response to show the result
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Upload Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f4f4f9;
            }}
            .container {{
                width: 100%;
                max-width: 500px;
                background: #fff;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 10px;
            }}
            h1 {{
                font-size: 24px;
                text-align: center;
                margin-bottom: 20px;
            }}
            p {{
                font-size: 18px;
                text-align: center;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CSV Processing Result</h1>
            <p>{result}</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
