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
      // Adjust the URL if your backend is hosted elsewhere
      const response = await axios.post("http://127.0.0.1:8000/upload/assessment", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const { assessments } = response.data;
      if (assessments && assessments.length > 0) {
        // Turn the array of objects into a readable string:
        const lines = assessments.map(a =>
          `${a.text} → ${a.is_correct ? "✅ true" : "❌ false"}`
        );
        setFeedback(lines.join("\n"));
        setErrorMessage("");
      } else {
        setErrorMessage("No equations found or extraction failed.");
      }
    } catch (error) {
      console.error("Upload error:", error);
      setErrorMessage("An error occurred during upload.");
    }
  };

  // Configure react-dropzone
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true, // prevent the entire area from triggering the file dialog on click
    noKeyboard: true // prevent ENTER/SPACE from triggering the file dialog
  });

  return (
    <div className="p-4 border-dashed border-2 border-gray-400 rounded-md">
      {/* The Drop Area */}
      <div {...getRootProps()} className="cursor-pointer py-10 text-center">
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the file here...</p>
        ) : (
          <p>Drag a PDF file here, or use the button below to upload.</p>
        )}
      </div>

      {/* A dedicated button to open the file dialog */}
      <div className="flex justify-center mt-2">
        <button
          type="button"
          onClick={open}
          className="bg-blue-600 text-white font-semibold py-2 px-4 rounded"
        >
          Choose File
        </button>
      </div>

      {/* Display extracted text or error */}
      {feedback && (
        <div className="mt-4 p-4 bg-gray-100 rounded shadow">
          <h3 className="font-bold mb-2">Extracted Text:</h3>
          <pre className="text-gray-700 whitespace-pre-wrap">
          {feedback}
          </pre>       
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
