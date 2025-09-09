import { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import FaceSlide from "./FaceSlide";
import ArmSlide from "./ArmSlide";
import SpeechSlide from "./SpeechSlide";
import EndSlide from "./EndSlide";
//import TopRightMenu from "./TopRightMenu";

// [ADD] 라우팅 훅 추가
import { useLocation, useNavigate, useParams } from "react-router-dom";

export default function TestCarousel() {
    const slides = [<FaceSlide key="face" />, <ArmSlide key="arm" />, <SpeechSlide key="speech" />, <EndSlide key="end" />];
    const [current, setCurrent] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    // [ADD] URL <-> 슬라이드 단계 매핑
    const STEPS = ["face", "arm", "speech", "end"];
    const navigate = useNavigate();
    const location = useLocation();
    const { step } = useParams(); // face | arm | speech | end | undefined

    const nextSlide = () => {
        if (isAnimating || current >= slides.length - 1) return;
        setIsAnimating(true);
        setCurrent((prev) => prev + 1);
    };

    const prevSlide = () => {
        if (isAnimating || current <= 0) return;
        setIsAnimating(true);
        setCurrent((prev) => prev - 1);
    };

    // ⏱️ 슬라이드 전환 애니메이션 타이머 (700ms 후 해제)
    useEffect(() => {
        if (!isAnimating) return;
        const timer = setTimeout(() => {
            setIsAnimating(false);
        }, 700);
        return () => clearTimeout(timer);
    }, [current]); // 기존 코드 유지

    // [ADD] 1) /test로만 들어오면 /test/face로 정규화
    useEffect(() => {
        if (location.pathname === "/test") {
            navigate("/test/face", { replace: true });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [location.pathname]);

    // [ADD] 2) URL이 바뀌면 슬라이드 인덱스를 맞춘다 (뒤/앞으로 가기 포함)
    useEffect(() => {
        const key = step ?? "face";
        const idx = Math.max(0, STEPS.indexOf(key));
        if (idx !== current) {
            setCurrent(idx);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [step]);

    // [ADD] 3) 슬라이드 인덱스가 내부에서 바뀌면 URL도 동기화
    useEffect(() => {
        const target = `/test/${STEPS[current]}`;
        if (location.pathname !== target) {
            navigate(target, { replace: false });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [current]);

    return (
        <div className="w-screen h-screen flex items-center justify-center bg-white relative overflow-hidden">
            <div className="w-[133vh] h-[133vh] rounded-full border-[7vw] border-[#f6f6f6] shadow-xl overflow-hidden flex items-center justify-center z-0 relative">
                <div
                    className="flex w-full h-full"
                    style={{
                        transform: `translateX(-${current * 100}%)`,
                        transition: "transform 700ms ease-in-out",
                    }}
                >
                    {slides.map((slide, idx) => (
                        <div key={idx} className="flex-shrink-0 w-full h-full">
                            {slide}
                        </div>
                    ))}
                </div>
            </div>

            {/* 왼쪽 화살표 */}
            {current > 0 && (
                <button
                    onClick={prevSlide}
                    className="absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-[calc(50%+calc(67.5px+60vh))]
                    w-[155px] h-[155px] rounded-full bg-white shadow-lg flex items-center justify-center z-50
                    hover:scale-110 transition-transform duration-300 active:scale-95"
                >
                    <ChevronLeft className="w-24 h-24 text-blue-600" strokeWidth={3} />
                </button>
            )}

            {/* 오른쪽 화살표 */}
            {current < slides.length - 1 && (
                <button
                    onClick={nextSlide}
                    className="absolute top-1/2 left-1/2 -translate-y-1/2 translate-x-[calc(50%+calc(67.5px+44vh))]
                    w-[155px] h-[155px] rounded-full bg-white shadow-lg flex items-center justify-center z-50
                    hover:scale-110 transition-transform duration-300 active:scale-95"
                >
                    <ChevronRight className="w-24 h-24 text-blue-600" strokeWidth={3} />
                </button>
            )}
        </div>
    );
}
