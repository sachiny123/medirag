class QueryEnhancer:
    def enhance(self, query: str) -> str:
        enhancements = {
            "rash": "skin rash dermatology allergic dermatitis viral rash fungal infection",
            "fever": "fever temperature pyrexia viral infection",
            "itchy": "itching pruritus allergic",
            "pain": "pain ache soreness inflammation",
            "cough": "cough respiratory infection",
            "headache": "headache migraine cephalalgia",
            "bleeding": "hemorrhage blood loss trauma",
            "stomach discomfort": "stomach discomfort abdominal irritation digestion problem mild stomach infection",
            "digestion issue": "stomach discomfort abdominal irritation digestion problem mild stomach infection"
        }
        enhanced_terms = []
        words = query.lower().split()
        for word in words:
            for key, val in enhancements.items():
                if key in word and val not in enhanced_terms:
                    enhanced_terms.append(val)
        
        if enhanced_terms:
            return query + " " + " ".join(enhanced_terms)
        return query
