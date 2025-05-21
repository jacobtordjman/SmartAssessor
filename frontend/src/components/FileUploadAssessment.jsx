// frontend/src/components/FileUploadAssessment.jsx

import { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";

export default function FileUploadAssessment() {
  const [feedback, setFeedback] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload/assessment",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      console.log("Backend JSON:", response.data);

      const { evaluation } = response.data;
      if (evaluation && evaluation.trim()) {
        setFeedback(evaluation);
        setErrorMessage("");
      } else {
        setErrorMessage("No evaluation returned from server.");
        setFeedback("");
      }
    } catch (error) {
      console.error("Upload error:", error);
      setErrorMessage("An error occurred during upload.");
      setFeedback("");
    }
  };

  const { getRootProps, getInputProps, isDragActive, open } =
    useDropzone({
      onDrop,
      noClick: true,
      noKeyboard: true,
    });

  return (
    <div className="p-4 border-dashed border-2 border-gray-400 rounded-md">
      <div {...getRootProps()} className="cursor-pointer py-10 text-center">
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the PDF here...</p>
        ) : (
          <p>Drag a PDF file here, or use the button below to upload.</p>
        )}
      </div>

      <div className="flex justify-center mt-2">
        <button
          type="button"
          onClick={open}
          className="bg-blue-600 text-white font-semibold py-2 px-4 rounded"
        >
          Choose File
        </button>
      </div>

      {feedback && (
        <div className="mt-4 p-4 bg-gray-100 rounded shadow">
          <h3 className="font-bold mb-2">Evaluation:</h3>
          <pre className="text-gray-700 whitespace-pre-wrap">{feedback}</pre>
        </div>
      )}

      {errorMessage && (
        <div className="mt-4 text-red-500">
          <p>{errorMessage}</p>
        </div>
      )}
    </div>
  );
}
