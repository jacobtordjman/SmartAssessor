import FileUploadAssessment from "./components/FileUploadAssessment";

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 flex items-center justify-center p-6">
      <div className="text-center max-w-2xl w-full">
        <h1 className="text-5xl font-extrabold text-indigo-700 mb-10 drop-shadow-lg animate-pulse">
          Smart Assessor
        </h1>
        <div className="bg-white rounded-3xl shadow-xl p-8 transition-all duration-300 hover:shadow-2xl">
          <FileUploadAssessment />
        </div>
      </div>
    </div>
  );
}
