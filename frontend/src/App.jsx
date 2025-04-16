import FileUploadAssessment from "./components/FileUploadAssessment";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
      <h1 className="text-3xl font-bold text-blue-600 mb-8">
        Feedback & Reinforcement App
      </h1>
      <FileUploadAssessment />
    </div>
  );
}
