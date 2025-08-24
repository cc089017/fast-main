// src/components/SpeechSlide.jsx
export default function SpeechSlide() {
    return (
        <div className="w-full h-full flex flex-col items-center justify-center text-center px-6">
            {/* 타이틀 */}
            <h1 className="text-9xl font-bold text-gray-400 mb-2">
                <span className="text-gray-400">F.A.</span><span className="text-blue-600">S</span><span className="text-gray-400">.T</span>
            </h1>
            <p className="text-3xl text-black mb-14">Speech Scan</p>

            {/* 중앙 내용 */}
            <div className="flex flex-row items-center justify-center gap-10 mb-20">
                {/* 캐릭터 이미지 */}
                <img
                    src="/speech.png" // ✅ public 경로 기준
                    alt="Face Guide"
                    className="w-[360px] h-auto"
                />

                {/* 검사 가이드 텍스트 */}
                <div className="text-left text-3xl text-gray-800 ml-0 leading-relaxed">
                    <p>1. 가이드라인 추가예정1</p>
                    <p className="mt-4">2. 가이드라인 추가예정2</p>
                    <p className="mt-4">3. 가이드라인 추가예정3</p>
                </div>
            </div>

            {/* 검사 시작 버튼 */}
            <button
                className="bg-blue-600 text-white text-4xl font-normal font-sans px-7 py-4 rounded-full shadow-lg
                   hover:scale-110 hover:bg-blue-700 hover:font-semibold hover:shadow-xl transition-transform duration-300"
            >
                검사 시작하기
            </button>
        </div>
    );
}
