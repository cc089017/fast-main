import { useParams, Link } from "react-router-dom";

export default function ResultDetail() {
    const { id } = useParams();
    return (
        <div className="min-h-screen bg-white px-6 py-10">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-4xl font-bold mb-6">검사 결과지 상세</h1>
                <p className="text-xl text-gray-700 mb-10">ID: {id}</p>

                <div className="p-6 rounded-2xl border">
                    <p className="text-gray-600">여기에 Face / Arm / Speech 세부 지표, 그래프, 녹화 스냅샷 등을 배치할 예정.</p>
                </div>

                <Link to="/results" className="inline-block mt-8 text-blue-600 hover:underline text-xl">
                    ← 목록으로
                </Link>
            </div>
        </div>
    );
}
