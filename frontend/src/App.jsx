import { useState } from "react";
import Navbar, { Features, Footer } from "./components/UI";
import UploadSection from "./components/UploadSection";
import ResultSection from "./components/ResultSection";
import "./App.css";

function App() {
  const [result, setResult] = useState(null);

  const handlePrediction = (data) => {
    setResult(data);
  };

  const handleReset = () => {
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-[#0a0f1c] text-white flex flex-col font-sans overflow-x-hidden selection:bg-[#818CF8]/30">
      <Navbar />

      {/* Decorative Background Elements */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-[#6366F1]/20 blur-[120px]" />
        <div className="absolute top-[20%] right-[-10%] w-[30%] h-[50%] rounded-full bg-[#C084FC]/10 blur-[140px]" />
        <div className="absolute bottom-[-10%] left-[20%] w-[50%] h-[40%] rounded-full bg-[#818CF8]/10 blur-[150px]" />
      </div>

      <main className="flex-1 w-full flex flex-col pt-32 relative z-10">
        {!result && (
          <div className="text-center px-4 mb-24 mt-8">
            <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight">
              Detect Brain Tumors with <br className="hidden md:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#818CF8] via-[#C084FC] to-[#F472B6]">
                Clinical Precision
              </span>
            </h1>
            <p className="text-[#94A3B8] text-lg md:text-xl max-w-2xl mx-auto font-medium">
              A state-of-the-art Convolutional Neural Network analyzing MRI scans
              to classify Glioma, Meningioma, Pituitary, or healthy brains in seconds.
            </p>
          </div>
        )}

        {result ? (
          <ResultSection result={result} onReset={handleReset} />
        ) : (
          <>
            <UploadSection onPrediction={handlePrediction} />
            <div className="mt-32">
              <Features />
            </div>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
