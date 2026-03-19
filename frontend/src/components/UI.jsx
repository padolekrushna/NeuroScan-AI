import { Brain, Shield, Clock, HeartPulse } from "lucide-react";

export default function Navbar() {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-white/5 bg-[#0a0f1c]/70">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#6366F1] to-[#9333EA] p-[1px]">
                        <div className="w-full h-full bg-[#0a0f1c] rounded-xl flex items-center justify-center">
                            <Brain className="w-6 h-6 text-[#A78BFA]" />
                        </div>
                    </div>
                    <span className="text-xl font-bold text-white tracking-tight">Neuro<span className="text-transparent bg-clip-text bg-gradient-to-r from-[#A78BFA] to-[#C084FC]">Scan</span> AI</span>
                </div>
                
                <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium">
                    <Shield className="w-4 h-4" />
                    <span>Clinical Grade Model (96% Acc)</span>
                </div>
            </div>
        </nav>
    );
}

export function Features() {
    const features = [
        {
            icon: <Brain className="w-6 h-6 text-[#818CF8]" />,
            title: "Advanced CNN",
            desc: "Custom-built convolutional layers trained on thousands of clinical MRI scans."
        },
        {
            icon: <Clock className="w-6 h-6 text-[#C084FC]" />,
            title: "Sub-second Analysis",
            desc: "Get highly accurate predictions and confidence scores in milliseconds."
        },
        {
            icon: <HeartPulse className="w-6 h-6 text-[#F472B6]" />,
            title: "Four-Class Detection",
            desc: "Accurately distinguishes between Glioma, Meningioma, Pituitary, and No Tumor."
        }
    ];

    return (
        <section className="relative z-10 max-w-7xl mx-auto px-4 py-24 border-t border-white/5">
            <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-white mb-4">Powered by Advanced Deep Learning</h2>
                <p className="text-[#94A3B8] max-w-2xl mx-auto">
                    Our architecture utilizes highly optimized convolutional layers to extract complex spatial features from medical images.
                </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-8">
                {features.map((f, i) => (
                    <div key={i} className="glass-card rounded-2xl p-8 hover:-translate-y-2 transition-transform duration-300">
                        <div className="w-14 h-14 rounded-xl bg-white/5 flex items-center justify-center mb-6">
                            {f.icon}
                        </div>
                        <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
                        <p className="text-[#94A3B8] text-sm leading-relaxed">{f.desc}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}

export function Footer() {
    return (
        <footer className="relative z-10 border-t border-white/5 py-12 text-center mt-auto">
            <p className="text-[#94A3B8] text-sm mb-4">
                Empowering Healthcare with Artificial Intelligence
            </p>
            <div className="flex justify-center gap-6 mb-8 text-sm">
                <a href="https://github.com/jaidatt007" className="text-[#64748B] hover:text-white transition-colors">GitHub</a>
                <a href="https://www.linkedin.com/in/jaidattkale/" className="text-[#64748B] hover:text-white transition-colors">LinkedIn</a>
            </div>
            <p className="text-[#475569] text-xs">
                © 2026 Brain Tumor Detection System. For research and educational purposes only.
            </p>
        </footer>
    );
}
