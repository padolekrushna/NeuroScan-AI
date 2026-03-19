import { useState, useRef } from "react";
import axios from "axios";
import { UploadCloud, CheckCircle, AlertTriangle, FileImage, X, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function UploadSection({ onPrediction }) {
    const [file, setFile] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState("");
    const fileInputRef = useRef(null);

    const handleFileSelect = (selectedFile) => {
        if (!selectedFile) return;

        if (!selectedFile.type.startsWith("image/")) {
            setError("Please select a valid image file (JPG, PNG).");
            return;
        }

        if (selectedFile.size > 10 * 1024 * 1024) {
            setError("File size exceeds 10MB limit.");
            return;
        }

        setFile(selectedFile);
        setError("");
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const droppedFile = e.dataTransfer.files[0];
        handleFileSelect(droppedFile);
    };

    const removeFile = () => {
        setFile(null);
        setError("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleSubmit = async () => {
        if (!file) return;

        setIsLoading(true);
        setError("");

        const formData = new FormData();
        formData.append("image", file);

        try {
            // Using a relative URL - Vite proxy or Vercel config will map this
            const apiUrl = import.meta.env.VITE_API_URL || "/api";
            console.log(`Making request to ${apiUrl}/predict`);
            
            const response = await axios.post(`${apiUrl}/predict`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            onPrediction({
                ...response.data,
                imageUrl: URL.createObjectURL(file), // create a local preview URL
            });
        } catch (err) {
            console.error("Prediction error:", err);
            setError(
                err.response?.data?.detail ||
                "Failed to analyze the MRI scan. Please make sure the backend is running."
            );
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <section className="relative z-10 max-w-2xl mx-auto -mt-16 w-full px-4">
            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="glass-card p-8 md:p-10 rounded-3xl"
            >
                <div className="text-center mb-8">
                    <h2 className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-[#818CF8] to-[#C084FC] mb-2">
                        Upload MRI Scan
                    </h2>
                    <p className="text-[#94A3B8]">
                        Our deep learning model will analyze the image to detect potential tumors.
                    </p>
                </div>

                <AnimatePresence mode="wait">
                    {!file ? (
                        <motion.div
                            key="upload-box"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer flex flex-col items-center justify-center ${
                                isDragging 
                                    ? "border-[#818CF8] bg-white/5" 
                                    : "border-[#334155] border-white/20 hover:border-[#818CF8]/50 hover:bg-white/5"
                            }`}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#818CF8]/20 to-[#C084FC]/20 flex items-center justify-center mb-4">
                                <UploadCloud className="w-8 h-8 text-[#A78BFA]" />
                            </div>
                            <h3 className="text-lg font-medium text-white mb-1">
                                Drag & drop your MRI image here
                            </h3>
                            <p className="text-sm text-[#94A3B8] mb-4">
                                or click to browse files
                            </p>
                            <p className="text-xs text-[#64748B]">
                                Supported formats: JPG, PNG, JPEG | Max size: 10MB
                            </p>
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={(e) => handleFileSelect(e.target.files?.[0])}
                                className="hidden"
                                accept="image/*"
                            />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="file-preview"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="bg-white/5 border border-white/10 rounded-2xl p-4 flex items-center gap-4 mb-6"
                        >
                            <div className="w-12 h-12 shrink-0 bg-[#818CF8]/20 rounded-xl flex items-center justify-center">
                                <FileImage className="w-6 h-6 text-[#818CF8]" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-white truncate">
                                    {file.name}
                                </p>
                                <p className="text-xs text-[#94A3B8]">
                                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                                </p>
                            </div>
                            <button
                                onClick={removeFile}
                                className="p-2 hover:bg-white/10 text-[#94A3B8] hover:text-red-400 rounded-lg transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>

                {error && (
                    <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3 text-red-400 text-sm"
                    >
                        <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                        <p>{error}</p>
                    </motion.div>
                )}

                <motion.button
                    onClick={handleSubmit}
                    disabled={!file || isLoading}
                    whileHover={file && !isLoading ? { scale: 1.02 } : {}}
                    whileTap={file && !isLoading ? { scale: 0.98 } : {}}
                    className={`w-full mt-6 py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all duration-300 ${
                        !file 
                            ? "bg-white/5 text-[#64748B] cursor-not-allowed border border-white/5" 
                            : isLoading
                                ? "bg-gradient-to-r from-[#4F46E5] to-[#7C3AED] opacity-80 cursor-wait text-white"
                                : "bg-gradient-to-r from-[#6366F1] to-[#9333EA] text-white shadow-[0_4px_20px_rgba(99,102,241,0.4)] hover:shadow-[0_4px_25px_rgba(147,51,234,0.5)]"
                    }`}
                >
                    {isLoading ? (
                        <>
                            <Activity className="w-5 h-5 animate-pulse" />
                            Analyzing Scan...
                        </>
                    ) : (
                        <>
                            <Activity className="w-5 h-5" />
                            Run Analysis
                        </>
                    )}
                </motion.button>
            </motion.div>
        </section>
    );
}
