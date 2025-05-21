import brainIcon from "../assets/brain.png";
import { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";


export default function Home() {
  const [feedback, setFeedback] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

    const onDrop = async (accepted) => {
    if (accepted.length === 0) return;
    const file = accepted[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/upload/assessment",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      console.log("Backend JSON:", res.data);

      const { evaluation } = res.data;

      // If the backend ever switches to objects, handle both shapes
      if (Array.isArray(evaluation)) {
        const lines = evaluation.map(
          (a) => `${a.text} → ${a.is_correct ? "✅ true" : "❌ false"}`
        );
        setFeedback(lines.join("\n"));
      } else if (typeof evaluation === "string" && evaluation.trim()) {
        setFeedback(evaluation);
      } else {
        setErrorMessage("No evaluation returned from server.");
        setFeedback("");
      }
    } catch (err) {
      console.error("Upload error:", err);
      setErrorMessage("An error occurred during upload.");
      setFeedback("");
    }
  };

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 flex flex-col items-center justify-start p-6 relative overflow-hidden text-center">
      <div className="absolute -top-32 -left-32 w-96 h-96 bg-purple-300 opacity-30 rounded-full filter blur-3xl animate-pulse"></div>
      <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-pink-300 opacity-30 rounded-full filter blur-3xl animate-pulse"></div>

      <div className="z-10 max-w-6xl w-full">
        <img src={brainIcon} alt="Smart Assessor Logo" style={{ width: "120px", height: "120px" }} className="mx-auto mb-2" />
        <h1 className="text-5xl font-extrabold text-indigo-700 mb-4 drop-shadow-lg animate-pulse">Smart Assessor</h1>
        <p className="text-lg text-gray-700 mb-12">
          Learning adventures await! Upload your PDF and let the fun begin.
        </p>

        <div className="card-row mb-16">
          <div className="bg-white rounded-2xl p-6 shadow-xl border border-gray-200 w-full max-w-sm mx-auto hover:scale-105 transition-all">
            <div className="w-16 h-16 mb-4 rounded-full bg-gradient-to-br from-purple-400 to-indigo-500 flex items-center justify-center text-white text-3xl shadow-md">
              📖
            </div>
            <h3 className="text-xl font-bold mb-2 text-indigo-800">Explore & Learn</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Dive into documents with tools that make learning interactive and engaging.
            </p>
          </div>
          

          <div className="bg-white rounded-2xl p-6 shadow-xl border border-gray-200 w-full max-w-sm mx-auto hover:scale-105 transition-all">
            <div className="w-16 h-16 mb-4 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center text-white text-3xl shadow-md">
              ✨
            </div>
            <h3 className="text-xl font-bold mb-2 text-indigo-800">Spark Curiosity</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Unlock insights and connect ideas with AI-powered discovery features.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-xl border border-gray-200 w-full max-w-sm mx-auto hover:scale-105 transition-all">
            <div className="w-16 h-16 mb-4 rounded-full bg-gradient-to-br from-green-400 to-teal-500 flex items-center justify-center text-white text-3xl shadow-md">
              🎖️
            </div>
            <h3 className="text-xl font-bold mb-2 text-indigo-800">Celebrate Wins</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Track your learning journey and earn cool badges for your achievements.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-8 transition-all duration-300 hover:shadow-2xl">
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
        </div>
      </div>
    </div>
  );
}
