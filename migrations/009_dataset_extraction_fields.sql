ALTER TABLE datasets ADD COLUMN modalities_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN osi_layers_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN collection_environment TEXT;
ALTER TABLE datasets ADD COLUMN known_users_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN availability_notes TEXT;
