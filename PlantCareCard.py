from pydantic import BaseModel

class PlantCareCard(BaseModel):
    common_name: str
    latin_name: str
    care_difficulty: str
    watering_frequency: str
    sunlight: str
    soil_type: str
    fertilizer: str
    outdoors: bool
    notes: str