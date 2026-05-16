import cv2
import numpy as np
import easyocr
import random
import re

from pydantic import Field


class Tools:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    # =========================================================
    # IMAGE PREPROCESSING
    # =========================================================
    def preprocess_image(
        self,
        image_path: str
    ):

        image = cv2.imread(image_path)

        if image is None:

            raise Exception(
                "Unable to read image"
            )

        h, w = image.shape[:2]

        # Auto rotate
        if h > w:

            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.fastNlMeansDenoising(
            gray
        )

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        sharpen = cv2.filter2D(
            gray,
            -1,
            kernel
        )

        processed_path = (
            image_path + "_processed.jpg"
        )

        cv2.imwrite(
            processed_path,
            sharpen
        )

        return processed_path
    def forgery_detection(self, image_path):
    
        try:
            import cv2
            import numpy as np

            # Read image
            image = cv2.imread(image_path)

            if image is None:
                return {
                    "status": "FAILED",
                    "error": "Unable to read image"
                }

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Noise analysis
            noise_score = np.std(gray)

            suspicious = False

            if noise_score > 70:
                suspicious = True

            edge_pixels = np.sum(edges > 0)

            forgery_score = float(min(edge_pixels / 100000, 1.0))

            return {
                "status": "SUCCESS",
                "forgery_detected": bool(suspicious),
                "forgery_score": forgery_score,
                "analysis": {
                    "noise_score": float(noise_score),
                    "edge_pixels": int(edge_pixels)
                }
            }

        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
    # =========================================================
    # OCR EXTRACTION
    # =========================================================
        # =========================================================
    # OCR EXTRACTION
    # =========================================================
    def extract_text_from_image(
        self,
        image_path: str
    ):

        results = self.reader.readtext(
            image_path,
            detail=1
        )

        cleaned = []

        noise_words = [

            "fel",
            "hrt",
            "tat",
            "rni",
            "aq1",
            "ayaha",
            "etatt",
            "now]",
            "#en",
            "31ecbr",
            "77aafty"
        ]

        for result in results:

            # ========================================
            # SAFE EASYOCR PARSING
            # ========================================

            if len(result) < 3:

                continue

            text = result[1]

            confidence = result[2]

            # Ignore low confidence OCR
            if confidence < 0.40:

                continue

            cleaned_text = text.strip()

            # Ignore tiny text
            if len(cleaned_text) < 3:

                continue

            # Ignore numeric garbage
            if cleaned_text.isnumeric():

                continue

            # Ignore symbol-heavy garbage
            if sum(
                c.isalpha()
                for c in cleaned_text
            ) < 2:

                continue

            lower_text = cleaned_text.lower()

            # Ignore OCR noise words
            if any(
                noise in lower_text
                for noise in noise_words
            ):

                continue

            cleaned.append(
                cleaned_text
            )

        return cleaned

    # =========================================================
    # KYC FIELD EXTRACTION
    # =========================================================
    def extract_kyc_fields(
        self,
        text
    ):

        combined = " ".join(text)

        combined_lower = combined.lower()

        aadhaar_number = None

        aadhaar_match = re.search(
            r"\d{4}\s\d{4}\s\d{4}",
            combined
        )

        if aadhaar_match:

            aadhaar_number = (
                aadhaar_match.group()
            )

        dob = None

        dob_match = re.search(
            r"\d{2}[\/\-\s]\d{2}[\/\-\s]\d{4}",
            combined
        )

        if dob_match:

            dob = dob_match.group()

        gender = None

        if "female" in combined_lower:

            gender = "Female"

        elif "male" in combined_lower:

            gender = "Male"

        ignore_words = [

            "government",
            "india",
            "aadhaar",
            "authority",
            "unique",
            "identification",
            "address",
            "dob",
            "male",
            "female",
            "proof",
            "citizenship"
        ]

        name = None

        for line in text:

            cleaned = line.strip()

            if len(cleaned) < 5:

                continue

            lower_line = cleaned.lower()

            if any(
                word in lower_line
                for word in ignore_words
            ):

                continue

            words = cleaned.split()

            if len(words) < 2:

                continue

            candidate = " ".join(words)

            if candidate.isupper():

                continue

            if candidate.istitle():

                name = candidate
                break

        return {

            "name":
                name,

            "dob":
                dob,

            "aadhaar_number":
                aadhaar_number,

            "gender":
                gender
        }

    # =========================================================
    # AUTOMATED KYC
    # =========================================================
    def automated_kyc_verification(
        self,
        image_path: str
    ):

        try:

            processed = self.preprocess_image(
                image_path
            )

            text = self.extract_text_from_image(
                processed
            )

            combined = " ".join(
                text
            ).lower()

            detected = []

            if (
                "aadhaar" in combined
                or "uidai" in combined
            ):

                detected.append(
                    "aadhaar"
                )

            if (
                "dob" in combined
                or "date of birth" in combined
            ):

                detected.append(
                    "dob"
                )

            structured = self.extract_kyc_fields(
                text
            )

            if structured["name"]:

                detected.append(
                    "name"
                )

            missing = list(

                set([
                    "name",
                    "dob",
                    "aadhaar"
                ]) - set(detected)
            )

            return {

                "status":
                    "SUCCESS",

                "structured_fields":
                    structured,

                "ocr_text":
                    text,

                "detected_fields":
                    detected,

                "missing_fields":
                    missing,

                "is_valid_document":
                    len(missing) == 0
            }

        except Exception as e:

            return {

                "status":
                    "FAILED",

                "error":
                    str(e)
            }

    # =========================================================
    # CHEQUE FRAUD DETECTION
    # =========================================================
    def cheque_fraud_detection(
        self,
        image_path: str
    ):

        try:

            processed = self.preprocess_image(
                image_path
            )

            text = self.extract_text_from_image(
                processed
            )

            combined = " ".join(text)

            combined_lower = combined.lower()

            bank_name = None

            if "axis bank" in combined_lower:

                bank_name = "Axis Bank"

            elif "state bank" in combined_lower:

                bank_name = "State Bank of India"

            elif "hdfc" in combined_lower:

                bank_name = "HDFC Bank"

            account_number = None

            account_match = re.search(
                r"\d{9,18}",
                combined
            )

            if account_match:

                account_number = (
                    account_match.group()
                )

            cheque_number = None

            cheque_match = re.search(
                r"\b\d{6}\b",
                combined
            )

            if cheque_match:

                cheque_number = (
                    cheque_match.group()
                )

            cancelled_keywords = [

                "cancelled",
                "cancel",
                "can_",
                "canc"
            ]

            is_cancelled = any(

                keyword in combined_lower

                for keyword in cancelled_keywords
            )

            signature_present = len(text) > 5

            fraud_score = 0.0

            if bank_name is None:
                fraud_score += 0.3

            if account_number is None:
                fraud_score += 0.3

            if not signature_present:
                fraud_score += 0.2

            if not is_cancelled:
                fraud_score += 0.1

            fraud_status = (
                "Cheque Looks Genuine"
            )

            if fraud_score > 0.5:

                fraud_status = (
                    "Potential Fraud Detected"
                )

            return {

                "status":
                    "SUCCESS",

                "cheque_details": {

                    "bank_name":
                        bank_name,

                    "account_number":
                        account_number,

                    "cheque_number":
                        cheque_number,

                    "cancelled_cheque":
                        is_cancelled,

                    "signature_present":
                        signature_present
                },

                "fraud_score":
                    fraud_score,

                "fraud_status":
                    fraud_status,

                "ocr_text":
                    text
            }

        except Exception as e:

            return {

                "status":
                    "FAILED",

                "error":
                    str(e)
            }

    # =========================================================
    # PAN FIELD EXTRACTION
    # =========================================================
    def extract_pan_fields(
        self,
        text
    ):

        combined = " ".join(text)

        pan_number = None

        pan_match = re.search(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            combined
        )

        if pan_match:

            pan_number = (
                pan_match.group()
            )

        dob = None

        dob_match = re.search(
            r"\d{2}/\d{2}/\d{4}",
            combined
        )

        if dob_match:

            dob = dob_match.group()

        name = None
        mother_name = None

        for i, line in enumerate(text):

            if "name" in line.lower():

                if i + 1 < len(text):

                    name = text[i + 1]

            if "mother" in line.lower():

                if i + 1 < len(text):

                    mother_name = text[i + 1]

        return {

            "pan_number":
                pan_number,

            "name":
                name,

            "mother_name":
                mother_name,

            "dob":
                dob
        }

    # =========================================================
    # DOCUMENT CLASSIFIER
    # =========================================================
    def document_classifier(
        self,
        image_path: str
    ):

        try:

            processed = self.preprocess_image(
                image_path
            )

            text = self.extract_text_from_image(
                processed
            )

            combined = " ".join(
                text
            ).lower()

            document_type = "Unknown"

            if (
                "aadhaar" in combined
                or "uidai" in combined
            ):

                document_type = "Aadhaar Card"

            elif (
                "income tax" in combined
                or "permanent account number"
                in combined
            ):

                document_type = "PAN Card"

            elif (
                "axis bank" in combined
                or "cheque" in combined
            ):

                document_type = "Cheque"

            elif "passport" in combined:

                document_type = "Passport"

            return {

                "status":
                    "SUCCESS",

                "document_type":
                    document_type,

                "ocr_text":
                    text
            }

        except Exception as e:

            return {

                "status":
                    "FAILED",

                "error":
                    str(e)
            }