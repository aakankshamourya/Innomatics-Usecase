from fastapi import FastAPI, UploadFile, File
import shutil
import os
import cv2
import numpy as np
from tools import Tools


app = FastAPI(
    title="Banking AI Platform"
)

tools = Tools()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# =========================================================
# SAVE FILE
# =========================================================
def save_upload_file(
    upload_file: UploadFile
):

    file_path = os.path.join(

        UPLOAD_DIR,

        upload_file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            upload_file.file,
            buffer
        )

    return file_path


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():

    return {

        "message":
            "Banking AI Platform Running"
    }


# =========================================================
# OCR TEST
# =========================================================
@app.post("/ocr")
def ocr_api(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    text = tools.extract_text_from_image(
        file_path
    )

    return {

        "status":
            "SUCCESS",

        "text":
            text
    }


# =========================================================
# KYC VERIFICATION
# =========================================================
@app.post("/kyc")
def kyc_verification(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.automated_kyc_verification(
        file_path
    )


# =========================================================
# CHEQUE FRAUD DETECTION
# =========================================================
@app.post("/cheque-fraud")
def cheque_fraud_detection(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.cheque_fraud_detection(
        file_path
    )


# =========================================================
# FORGERY DETECTION
# =========================================================
@app.post("/forgery")
def forgery_detection(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.forgery_detection(
        file_path
    )


# =========================================================
# LOAN DOCUMENT INTELLIGENCE
# =========================================================
@app.post("/loan-intelligence")
def loan_document_intelligence(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.loan_document_intelligence(
        file_path
    )


# =========================================================
# BANK STATEMENT ANALYZER
# =========================================================
@app.post("/bank-statement")
def bank_statement_analyzer(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.bank_statement_analyzer(
        file_path
    )


# =========================================================
# DOCUMENT CLASSIFIER
# =========================================================
@app.post("/document-classifier")
def document_classifier(
    file: UploadFile = File(...)
):

    file_path = save_upload_file(
        file
    )

    return tools.document_classifier(
        file_path
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