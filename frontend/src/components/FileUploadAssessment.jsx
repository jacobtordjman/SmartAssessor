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
      const response = await axios.post("http://127.0.0.1:8000/upload/assessment", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const { assessments } = response.data;
      if (assessments && assessments.length > 0) {
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

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true
  });

  return (
    <div className="p-6 border-4 border-dashed border-indigo-300 rounded-2xl bg-indigo-50 hover:bg-indigo-100 transition-colors">
      <div {...getRootProps()} className="cursor-pointer py-12 text-center text-indigo-700 font-medium">
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the file here...</p>
        ) : (
          <p>Drag a PDF file here, or use the button below to upload.</p>
        )}
      </div>

      <div className="flex justify-center mt-4">
        <button
          type="button"
          onClick={open}
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-6 rounded-full shadow-md transition-transform hover:scale-105"
        >
          Choose File
        </button>
      </div>

      {feedback && (
        <div className="mt-6 p-6 bg-white border border-gray-200 rounded-xl shadow-sm">
          <h3 className="font-semibold text-indigo-800 mb-2 text-lg">Results Overview:</h3>
          <pre className="text-gray-700 whitespace-pre-wrap">
            {feedback}
          </pre>
        </div>
      )}
      {errorMessage && (
        <div className="mt-4 text-red-500 font-medium">
          <p>{errorMessage}</p>
        </div>
      )}
    </div>
  );
}
