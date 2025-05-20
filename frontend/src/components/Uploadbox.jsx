export default function UploadBox() {
  return (
    <div className="bg-gradient-to-br from-purple-200 via-indigo-200 to-pink-200 rounded-3xl shadow-lg p-8 w-full max-w-3xl mx-auto transition hover:scale-[1.02]">
      <h2 className="text-2xl font-bold text-indigo-800 mb-6 flex items-center gap-2">
        <span className="text-3xl">☁️</span>
        Upload Your Learning Material
      </h2>

      <div className="bg-purple-200/30 rounded-2xl p-10 text-center text-white backdrop-blur shadow-inner">
        <div className="bg-gradient-to-tr from-orange-400 to-pink-500 rounded-xl w-16 h-16 mx-auto flex items-center justify-center mb-4 shadow-md">
          <span className="text-3xl">☁️</span>
        </div>
        <p className="text-xl font-semibold">Drag & Drop Your PDF Here</p>
        <p className="text-sm text-white/90 mt-1">or click to select your file</p>
      </div>

      <button
        className="mt-8 bg-gradient-to-r from-pink-500 to-orange-400 text-white font-semibold px-6 py-3 rounded-xl shadow hover:brightness-110 transition flex items-center justify-center gap-2 mx-auto"
      >
        ✨ Start My Learning Quest!
      </button>
    </div>
  );
}
