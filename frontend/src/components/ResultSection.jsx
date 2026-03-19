import { motion } from "framer-motion";
import { Brain, CheckCircle, AlertTriangle, ArrowLeft, Download } from "lucide-react";

export default function ResultSection({ result, onReset }) {
    const { prediction, confidence, probabilities, imageUrl } = result;

    const isTumor = prediction.toLowerCase() !== "no tumor";

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { staggerChildren: 0.1 }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
    };

    return (
        <section className="relative z-10 max-w-5xl mx-auto w-full px-4 pb-20 pt-10">
            <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="mb-8"
            >
                <button 
                    onClick={onReset}
                    className="flex items-center gap-2 text-[#94A3B8] hover:text-white transition-colors group"
                >
                    <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                    Back to Upload
                </button>
            </motion.div>

            <motion.div 
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="grid md:grid-cols-2 gap-8"
            >
                {/* Image Card */}
                <motion.div variants={itemVariants} className="glass-card rounded-3xl p-6 md:p-8 flex flex-col">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-10 h-10 rounded-lg bg-[#818CF8]/20 flex items-center justify-center">
                            <Brain className="w-5 h-5 text-[#818CF8]" />
                        </div>
                        <h3 className="text-xl font-semibold text-white">Analyzed Scan</h3>
                    </div>
                    
                    <div className="relative rounded-2xl overflow-hidden shadow-[0_8px_30px_rgba(0,0,0,0.3)] border border-white/10 flex-1 min-h-[300px] bg-black/40">
                        {imageUrl && (
                            <img 
                                src={imageUrl} 
                                alt="MRI Scan" 
                                className="absolute inset-0 w-full h-full object-contain p-2"
                            />
                        )}
                        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                            <span className="text-xs font-medium text-white/80 bg-black/40 px-3 py-1 rounded-full backdrop-blur-md">
                                Original MRI Image
                            </span>
                        </div>
                    </div>
                </motion.div>

                {/* Prediction Results Card */}
                <motion.div variants={itemVariants} className="glass-card rounded-3xl p-6 md:p-8">
                    <h3 className="text-xl font-semibold text-white mb-6">Diagnosis Results</h3>
                    
                    {/* Primary Result */}
                    <div className={`rounded-2xl p-6 border mb-8 ${
                        isTumor 
                            ? "bg-red-500/10 border-red-500/20 shadow-[inset_0_0_40px_rgba(239,68,68,0.05)]" 
                            : "bg-emerald-500/10 border-emerald-500/20 shadow-[inset_0_0_40px_rgba(16,185,129,0.05)]"
                    }`}>
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-[#94A3B8] text-sm uppercase tracking-wider mb-1 font-medium">
                                    Primary Detection
                                </p>
                                <h4 className={`text-4xl font-bold bg-clip-text text-transparent ${
                                    isTumor ? "bg-gradient-to-r from-red-400 to-rose-400" : "bg-gradient-to-r from-emerald-400 to-teal-400"
                                }`}>
                                    {prediction}
                                </h4>
                            </div>
                            <div className={`p-4 rounded-full ${isTumor ? "bg-red-500/20" : "bg-emerald-500/20"}`}>
                                {isTumor ? (
                                    <AlertTriangle className={`w-8 h-8 ${isTumor ? "text-red-400" : ""}`} />
                                ) : (
                                    <CheckCircle className="w-8 h-8 text-emerald-400" />
                                )}
                            </div>
                        </div>
                        
                        <div className="mt-8">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-white/80 font-medium">Confidence Score</span>
                                <span className="text-xl font-bold text-white">{confidence}%</span>
                            </div>
                            <div className="h-3 w-full bg-black/40 rounded-full overflow-hidden border border-white/5">
                                <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${confidence}%` }}
                                    transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
                                    className={`h-full relative overflow-hidden ${
                                        isTumor ? "bg-gradient-to-r from-red-500 to-rose-500" : "bg-gradient-to-r from-emerald-500 to-teal-500"
                                    }`}
                                >
                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full animate-[shimmer_2s_infinite]" />
                                </motion.div>
                            </div>
                        </div>
                    </div>

                    {/* Detailed Probabilities */}
                    <div className="space-y-4">
                        <h5 className="text-sm uppercase tracking-wider text-[#94A3B8] font-semibold mb-4">Detailed Probabilities</h5>
                        {Object.entries(probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([name, prob], i) => (
                            <div key={name} className="flex items-center gap-4">
                                <span className="text-sm font-medium text-white/90 w-28 truncate">{name}</span>
                                <div className="flex-1 h-1.5 bg-black/30 rounded-full overflow-hidden">
                                    <motion.div 
                                        initial={{ width: 0 }}
                                        animate={{ width: `${prob}%` }}
                                        transition={{ duration: 1, delay: 0.8 + i * 0.1 }}
                                        className="h-full bg-gradient-to-r from-[#818CF8] to-[#C084FC]"
                                    />
                                </div>
                                <span className="text-sm text-[#94A3B8] w-12 text-right">{prob}%</span>
                            </div>
                        ))}
                    </div>
                </motion.div>
            </motion.div>

            {/* Disclaimer */}
            <motion.div variants={itemVariants} className="mt-8 glass-card rounded-2xl p-6 border-l-4 border-l-[#818CF8] flex items-start gap-4">
                <AlertTriangle className="w-6 h-6 text-[#818CF8] shrink-0 mt-0.5" />
                <div>
                    <h4 className="text-white font-medium mb-1">Medical Disclaimer</h4>
                    <p className="text-[#94A3B8] text-sm leading-relaxed">
                        This AI-powered analysis is designed to assist healthcare professionals and should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult with qualified medical professionals for proper evaluation.
                    </p>
                </div>
            </motion.div>
        </section>
    );
}
