from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import traceback
import cv2
import numpy as np
from tools import Tools


# =========================================================
# INITIALIZE
# =========================================================

app = FastAPI(
    title="AI Banking MCP Server",
    version="1.0.0"
)

tools = Tools()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)
def save_upload(file):
    
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path

# =========================================================
# ALLOW NGROK / PUBLIC HOSTS
# =========================================================

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# =========================================================
# COMMON FILE SAVE FUNCTION
# =========================================================

def save_uploaded_file(
    file: UploadFile
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return file_path


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return JSONResponse(
        content={
            "status": "SUCCESS",
            "message": "AI Banking MCP Server Running"
        }
    )


# =========================================================
# MCP ENDPOINT
# =========================================================

@app.get("/mcp")
async def mcp():

    return JSONResponse(
        content={
            "name": "AI Banking MCP Server",
            "version": "1.0.0",
            "protocol": "MCP",
            "status": "ACTIVE"
        }
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return JSONResponse(
        content={
            "health": "ok"
        }
    )


# =========================================================
# OCR EXTRACTION
# =========================================================

@app.post("/ocr")
async def ocr_api(
    file: UploadFile = File(...)
):

    try:

        file_path = save_uploaded_file(
            file
        )

        processed = tools.preprocess_image(
            file_path
        )

        text = tools.extract_text_from_image(
            processed
        )

        return JSONResponse(
            content={
                "status": "SUCCESS",
                "ocr_text": text
            }
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "error": str(e)
            }
        )


# =========================================================
# AUTOMATED KYC VERIFICATION
# =========================================================

@app.post("/kyc")
async def kyc_api(
    file: UploadFile = File(...)
):

    try:

        file_path = save_uploaded_file(
            file
        )

        result = tools.automated_kyc_verification(
            file_path
        )

        return JSONResponse(
            content=result
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "error": str(e)
            }
        )


# =========================================================
# CHEQUE FRAUD DETECTION
# =========================================================

@app.post("/cheque")
async def cheque_api(
    file: UploadFile = File(...)
):

    try:

        file_path = save_uploaded_file(
            file
        )

        result = tools.cheque_fraud_detection(
            file_path
        )

        return JSONResponse(
            content=result
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "error": str(e)
            }
        )


# =========================================================
# DOCUMENT CLASSIFIER
# =========================================================

@app.post("/document_classifier")
async def document_classifier_api(
    file: UploadFile = File(...)
):

    try:

        file_path = save_uploaded_file(
            file
        )

        result = tools.document_classifier(
            file_path
        )

        return JSONResponse(
            content=result
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "error": str(e)
            }
        )


# =========================================================
# PAN EXTRACTION
# =========================================================

@app.post("/pan_extraction")
async def pan_extraction_api(
    file: UploadFile = File(...)
):

    try:

        file_path = save_uploaded_file(
            file
        )

        processed = tools.preprocess_image(
            file_path
        )

        text = tools.extract_text_from_image(
            processed
        )

        result = tools.extract_pan_fields(
            text
        )

        return JSONResponse(
            content={
                "status": "SUCCESS",
                "structured_fields": result,
                "ocr_text": text
            }
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "error": str(e)
            }
        )


# =========================================================
# SIGNATURE VERIFICATION
# =========================================================

@app.post("/signature_verification")
async def signature_verification(

    reference_signature: UploadFile,
    uploaded_signature: UploadFile
):

    try:

        # ============================================
        # SAVE FILES
        # ============================================

        ref_path = (
            f"temp_{reference_signature.filename}"
        )

        upload_path = (
            f"temp_{uploaded_signature.filename}"
        )

        with open(ref_path, "wb") as buffer:

            shutil.copyfileobj(
                reference_signature.file,
                buffer
            )

        with open(upload_path, "wb") as buffer:

            shutil.copyfileobj(
                uploaded_signature.file,
                buffer
            )

        # ============================================
        # READ IMAGES
        # ============================================

        ref_img = cv2.imread(
            ref_path,
            0
        )

        upload_img = cv2.imread(
            upload_path,
            0
        )

        if ref_img is None:

            raise Exception(
                "Reference signature image not readable"
            )

        if upload_img is None:

            raise Exception(
                "Uploaded signature image not readable"
            )

        # ============================================
        # RESIZE
        # ============================================

        ref_img = cv2.resize(
            ref_img,
            (300, 100)
        )

        upload_img = cv2.resize(
            upload_img,
            (300, 100)
        )

        # ============================================
        # DIFFERENCE SCORE
        # ============================================

        difference = cv2.absdiff(
            ref_img,
            upload_img
        )

        score = np.sum(
            difference
        ) / 1000000

        match = bool(score < 5)

        return {

            "status":
                "SUCCESS",

            "signature_match":
                match,

            "difference_score":
                float(score)
        }

    except Exception as e:

        return {

            "status":
                "FAILED",

            "error":
                str(e)
        }

# =========================================================
# FORGERY DETECTION
# =========================================================

@app.post("/forgery_detection")
async def forgery_detection_api(
    file: UploadFile = File(...)
):

    file_path = save_upload(file)

    result = tools.forgery_detection(file_path)

    return result
# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )