import os
from datetime import date, datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from ml_service import ml_service, resource_priority_service


BASE_DIR = Path(__file__).resolve().parent


def get_data_root() -> Path:
	"""
	Use a writable directory in Render when available.
	Defaults to the project directory for local development.
	"""
	render_disk_path = os.getenv("RENDER_DISK_PATH")
	if render_disk_path:
		return Path(render_disk_path).resolve()
	return BASE_DIR


DATA_ROOT = get_data_root()
DISASTERS_ROOT = DATA_ROOT / "Disasters"

# Initialize SQLAlchemy
db = SQLAlchemy()

def slugify(text: str) -> str:
	return "-".join(
		"".join(ch for ch in text.lower() if ch.isalnum() or ch in [' ', '-']).split()
	)


def ensure_disaster_folder(name: str, start_date: date, did: int) -> str:
	slug = slugify(name)
	folder_name = f"{start_date.strftime('%Y')}-{slug}-{did}"
	path = DISASTERS_ROOT / folder_name
	(path / 'intake').mkdir(parents=True, exist_ok=True)
	(path / 'shipments').mkdir(parents=True, exist_ok=True)
	(path / 'distributions').mkdir(parents=True, exist_ok=True)
	(path / 'attachments').mkdir(parents=True, exist_ok=True)
	return str(path)


def create_app():
	app = Flask(__name__)
	load_dotenv()
	app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
	
	# Configure database (DATABASE_URL preferred for production)
	default_sqlite_path = DATA_ROOT / "drms.db"
	database_url = os.getenv('DATABASE_URL', f'sqlite:///{default_sqlite_path}?check_same_thread=False')
	if database_url.startswith("postgres://"):
		database_url = database_url.replace("postgres://", "postgresql://", 1)
	app.config['SQLALCHEMY_DATABASE_URI'] = database_url
	app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	engine_options = {
		'pool_pre_ping': True,
		'pool_recycle': 300,
	}
	if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite:///'):
		engine_options['connect_args'] = {
			'timeout': 20,
			'check_same_thread': False
		}
	app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_options
	
	# Initialize database
	db.init_app(app)
	
	DISASTERS_ROOT.mkdir(exist_ok=True)
	
	# Create tables
	with app.app_context():
		db.create_all()
	
	register_routes(app)
	return app


# Database Models - Multi-Tenant Architecture
class Disaster(db.Model):
	__tablename__ = 'disaster'
	
	DisasterID = db.Column(db.Integer, primary_key=True)
	Name = db.Column(db.String(200), nullable=False)
	Type = db.Column(db.String(100))
	Location = db.Column(db.String(200))
	StartDate = db.Column(db.Date, nullable=False)
	EndDate = db.Column(db.Date)
	Severity = db.Column(db.String(50))
	FolderPath = db.Column(db.String(500), nullable=False)
	CreatedAt = db.Column(db.DateTime, default=datetime.utcnow)
	UpdatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	def __repr__(self):
		return f'<Disaster {self.DisasterID}: {self.Name}>'


class ResourceRequest(db.Model):
	"""Stores incoming resource requests with AI-assigned priority."""
	__tablename__ = 'resource_request'

	RequestID = db.Column(db.Integer, primary_key=True)
	DisasterID = db.Column(db.Integer, db.ForeignKey('disaster.DisasterID'), nullable=False)
	SeverityLevel = db.Column(db.String(20), nullable=False)
	PeopleAffected = db.Column(db.Integer, nullable=False)
	ResourceType = db.Column(db.String(50), nullable=False)
	LocationUrgency = db.Column(db.String(20), nullable=True)
	PredictedPriority = db.Column(db.String(20), nullable=False)
	CreatedAt = db.Column(db.DateTime, default=datetime.utcnow)

	disaster = db.relationship('Disaster', backref=db.backref('resource_requests', lazy=True))

	def __repr__(self):
		return f'<ResourceRequest {self.RequestID}: {self.PredictedPriority}>'


# Model cache to avoid SQLAlchemy conflicts
_tenant_model_cache = {}

# Dynamic model creation for disaster-specific tables
def create_tenant_models(disaster_id):
	"""Create disaster-specific table models dynamically with caching"""
	prefix = f"disaster_{disaster_id}"
	
	# Check if models are already cached
	if prefix in _tenant_model_cache:
		return _tenant_model_cache[prefix]
	
	# Create unique class names to avoid conflicts
	warehouse_class_name = f"TenantWarehouse_{disaster_id}"
	relief_item_class_name = f"TenantReliefItem_{disaster_id}"
	beneficiary_class_name = f"TenantBeneficiary_{disaster_id}"
	distribution_class_name = f"TenantDistribution_{disaster_id}"
	
	# Create the model classes
	TenantWarehouse = type(warehouse_class_name, (db.Model,), {
		'__tablename__': f'{prefix}_warehouse',
		'__module__': __name__,
		
		'WarehouseID': db.Column(db.Integer, primary_key=True),
		'Location': db.Column(db.String(200), nullable=False),
		'Capacity': db.Column(db.Integer, nullable=False, default=0),
		'CreatedAt': db.Column(db.DateTime, default=datetime.utcnow),
		
		'__repr__': lambda self: f'<TenantWarehouse {self.WarehouseID}: {self.Location}>'
	})
	
	TenantReliefItem = type(relief_item_class_name, (db.Model,), {
		'__tablename__': f'{prefix}_relief_item',
		'__module__': __name__,
		
		'ItemID': db.Column(db.Integer, primary_key=True),
		'WarehouseID': db.Column(db.Integer, db.ForeignKey(f'{prefix}_warehouse.WarehouseID'), nullable=False),
		'Name': db.Column(db.String(150), nullable=False),
		'Category': db.Column(db.String(100)),
		'Quantity': db.Column(db.Integer, nullable=False, default=0),
		'CreatedAt': db.Column(db.DateTime, default=datetime.utcnow),
		
		'__repr__': lambda self: f'<TenantReliefItem {self.ItemID}: {self.Name} ({self.Quantity})>'
	})
	
	TenantBeneficiary = type(beneficiary_class_name, (db.Model,), {
		'__tablename__': f'{prefix}_beneficiary',
		'__module__': __name__,
		
		'BeneficiaryID': db.Column(db.Integer, primary_key=True),
		'Name': db.Column(db.String(200), nullable=False),
		'Location': db.Column(db.String(200)),
		'Contact': db.Column(db.String(100)),
		'CreatedAt': db.Column(db.DateTime, default=datetime.utcnow),
		
		'__repr__': lambda self: f'<TenantBeneficiary {self.BeneficiaryID}: {self.Name}>'
	})
	
	TenantDistribution = type(distribution_class_name, (db.Model,), {
		'__tablename__': f'{prefix}_distribution',
		'__module__': __name__,
		
		'DistID': db.Column(db.Integer, primary_key=True),
		'BeneficiaryID': db.Column(db.Integer, db.ForeignKey(f'{prefix}_beneficiary.BeneficiaryID'), nullable=False),
		'ItemID': db.Column(db.Integer, db.ForeignKey(f'{prefix}_relief_item.ItemID'), nullable=False),
		'Quantity': db.Column(db.Integer, nullable=False),
		'Date': db.Column(db.Date, nullable=False),
		'CreatedAt': db.Column(db.DateTime, default=datetime.utcnow),
		
		'__repr__': lambda self: f'<TenantDistribution {self.DistID}: {self.Quantity} items>'
	})
	
	# Add relationships after class creation
	TenantReliefItem.warehouse = db.relationship(TenantWarehouse, backref=db.backref('items', lazy=True))
	TenantDistribution.beneficiary = db.relationship(TenantBeneficiary, backref=db.backref('distributions', lazy=True))
	TenantDistribution.item = db.relationship(TenantReliefItem, backref=db.backref('distributions', lazy=True))
	
	# Cache the models
	_tenant_model_cache[prefix] = (TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution)
	
	return TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution


def create_tenant_tables(disaster_id):
	"""Create tables for a specific disaster with proper error handling"""
	import time
	from sqlalchemy.exc import OperationalError
	
	TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = create_tenant_models(disaster_id)
	
	# Retry mechanism for table creation to handle database locks
	max_retries = 3
	retry_delay = 0.5
	
	for attempt in range(max_retries):
		try:
			# Create all tables for this disaster with explicit checkfirst
			TenantWarehouse.__table__.create(db.engine, checkfirst=True)
			TenantReliefItem.__table__.create(db.engine, checkfirst=True)
			TenantBeneficiary.__table__.create(db.engine, checkfirst=True)
			TenantDistribution.__table__.create(db.engine, checkfirst=True)
			
			# If we get here, all tables were created successfully
			break
			
		except OperationalError as e:
			if "database is locked" in str(e).lower() and attempt < max_retries - 1:
				print(f"Database locked, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
				time.sleep(retry_delay)
				retry_delay *= 2  # Exponential backoff
				continue
			else:
				# Re-raise the error if it's not a lock issue or we've exhausted retries
				raise e
	
	return TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution


def get_tenant_models(disaster_id):
	"""Get the model classes for a specific disaster"""
	return create_tenant_models(disaster_id)


def check_tenant_tables_exist(disaster_id):
	"""Check if tables for a disaster already exist"""
	from sqlalchemy import inspect
	
	inspector = inspect(db.engine)
	existing_tables = inspector.get_table_names()
	
	prefix = f"disaster_{disaster_id}"
	required_tables = [
		f"{prefix}_warehouse",
		f"{prefix}_relief_item", 
		f"{prefix}_beneficiary",
		f"{prefix}_distribution"
	]
	
	return all(table in existing_tables for table in required_tables)


def register_routes(app: Flask) -> None:
	@app.route('/')
	def index():
		disasters = Disaster.query.order_by(Disaster.StartDate.desc()).all()
		return render_template('index.html', disasters=disasters)

	@app.route('/disasters/new', methods=['GET', 'POST'])
	def create_disaster():
		if request.method == 'POST':
			try:
				# Get form data
				name = request.form.get('name', '').strip()
				type_ = request.form.get('type', '').strip()
				location = request.form.get('location', '').strip()
				start_date_str = request.form.get('start_date', '').strip()
				end_date_str = request.form.get('end_date', '').strip()
				severity = request.form.get('severity', '').strip()
				
				# Validate required fields
				if not name or not start_date_str:
					flash('Name and Start Date are required.', 'danger')
					return redirect(url_for('create_disaster'))
				
				# Parse dates
				try:
					start_date = date.fromisoformat(start_date_str)
				except ValueError:
					flash('Invalid Start Date format (YYYY-MM-DD).', 'danger')
					return redirect(url_for('create_disaster'))
				
				end_date = date.fromisoformat(end_date_str) if end_date_str else None
				
				# Begin transaction
				disaster = Disaster(
					Name=name,
					Type=type_,
					Location=location,
					StartDate=start_date,
					EndDate=end_date,
					Severity=severity,
					FolderPath=''  # Will be updated after ID generation
				)
				
				db.session.add(disaster)
				db.session.flush()  # Generate ID without committing
				
				# Create folder structure
				folder_path = ensure_disaster_folder(name, start_date, disaster.DisasterID)
				disaster.FolderPath = folder_path
				
				# Commit the disaster record first
				db.session.commit()
				
				# Create disaster-specific tables after committing the main record
				# This reduces the chance of database locks
				try:
					# Only create tables if they don't already exist
					if not check_tenant_tables_exist(disaster.DisasterID):
						create_tenant_tables(disaster.DisasterID)
					else:
						print(f"Tables for disaster {disaster.DisasterID} already exist, skipping creation")
				except Exception as table_error:
					# If table creation fails, we still have the disaster record
					# but we should inform the user
					flash(f'Disaster "{name}" created but table setup failed: {str(table_error)}', 'warning')
					return redirect(url_for('index'))
				
				flash(f'Disaster "{name}" created successfully with database tables!', 'success')
				return redirect(url_for('index'))
				
			except Exception as e:
				db.session.rollback()
				flash(f'Error creating disaster: {str(e)}', 'danger')
				return redirect(url_for('create_disaster'))
		
		return render_template('disaster_form.html')

	@app.route('/disasters/<int:disaster_id>')
	def disaster_detail(disaster_id: int):
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get warehouses for this disaster
		warehouses = TenantWarehouse.query.all()
		
		# Get items with warehouse info
		items = db.session.query(TenantReliefItem, TenantWarehouse).join(
			TenantWarehouse, TenantReliefItem.WarehouseID == TenantWarehouse.WarehouseID
		).all()
		
		# Get distributions with related info
		distributions = db.session.query(TenantDistribution, TenantBeneficiary, TenantReliefItem).join(
			TenantBeneficiary, TenantDistribution.BeneficiaryID == TenantBeneficiary.BeneficiaryID
		).join(
			TenantReliefItem, TenantDistribution.ItemID == TenantReliefItem.ItemID
		).order_by(TenantDistribution.Date.desc()).all()

		# Show latest resource requests with auto-assigned priorities
		resource_requests = ResourceRequest.query.filter_by(DisasterID=disaster_id).order_by(
			ResourceRequest.CreatedAt.desc()
		).limit(10).all()
		
		return render_template('disaster_detail.html', 
			disaster=disaster, 
			warehouses=warehouses, 
			items=items, 
			distributions=distributions,
			resource_requests=resource_requests)

	@app.route('/disasters/<int:disaster_id>/resources/new', methods=['GET', 'POST'])
	def add_resources(disaster_id: int):
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		if request.method == 'POST':
			try:
				# Get form data
				location = request.form.get('warehouse_location', '').strip()
				capacity = int(request.form.get('warehouse_capacity', '0') or 0)
				item_name = request.form.get('item_name', '').strip()
				item_cat = request.form.get('item_category', '').strip()
				item_qty = int(request.form.get('item_quantity', '0') or 0)
				
				# Validate required fields
				if not location or not item_name or item_qty <= 0:
					flash('Warehouse location, item name and positive quantity are required.', 'danger')
					return redirect(url_for('add_resources', disaster_id=disaster_id))
				
				# Begin transaction
				warehouse = TenantWarehouse(
					Location=location,
					Capacity=max(0, capacity)
				)
				
				db.session.add(warehouse)
				db.session.flush()  # Generate warehouse ID
				
				item = TenantReliefItem(
					WarehouseID=warehouse.WarehouseID,
					Name=item_name,
					Category=item_cat,
					Quantity=item_qty
				)
				
				db.session.add(item)
				db.session.commit()
				
				flash('Resources added successfully to disaster-specific tables!', 'success')
				return redirect(url_for('disaster_detail', disaster_id=disaster_id))
				
			except Exception as e:
				db.session.rollback()
				flash(f'Error adding resources: {str(e)}', 'danger')
				return redirect(url_for('add_resources', disaster_id=disaster_id))
		
		return render_template('resource_form.html', disaster=disaster)

	@app.route('/disasters/<int:disaster_id>/distribute', methods=['GET', 'POST'])
	def distribute(disaster_id: int):
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get available items for this disaster
		items = db.session.query(TenantReliefItem).join(
			TenantWarehouse, TenantReliefItem.WarehouseID == TenantWarehouse.WarehouseID
		).all()
		
		if request.method == 'POST':
			try:
				# Get form data
				beneficiary_name = request.form.get('beneficiary_name', '').strip()
				beneficiary_loc = request.form.get('beneficiary_location', '').strip()
				beneficiary_contact = request.form.get('beneficiary_contact', '').strip()
				item_id = int(request.form.get('item_id'))
				qty = int(request.form.get('quantity', '0') or 0)
				when_str = request.form.get('date')
				when = date.fromisoformat(when_str) if when_str else date.today()
				
				# Validate required fields
				if not beneficiary_name or qty <= 0:
					flash('Beneficiary name and positive quantity are required.', 'danger')
					return redirect(url_for('distribute', disaster_id=disaster_id))
				
				# Check item availability
				item = TenantReliefItem.query.get_or_404(item_id)
				if item.Quantity < qty:
					flash('Not enough inventory for the selected item.', 'danger')
					return redirect(url_for('distribute', disaster_id=disaster_id))
				
				# Begin transaction
				beneficiary = TenantBeneficiary(
					Name=beneficiary_name,
					Location=beneficiary_loc,
					Contact=beneficiary_contact
				)
				
				db.session.add(beneficiary)
				db.session.flush()  # Generate beneficiary ID
				
				distribution = TenantDistribution(
					BeneficiaryID=beneficiary.BeneficiaryID,
					ItemID=item.ItemID,
					Quantity=qty,
					Date=when
				)
				
				db.session.add(distribution)
				
				# Update inventory
				item.Quantity = max(0, item.Quantity - qty)
				
				db.session.commit()
				
				flash('Distribution recorded and inventory updated in disaster-specific tables!', 'success')
				return redirect(url_for('disaster_detail', disaster_id=disaster_id))
				
			except Exception as e:
				db.session.rollback()
				flash(f'Error recording distribution: {str(e)}', 'danger')
				return redirect(url_for('distribute', disaster_id=disaster_id))
		
		return render_template('distribution_form.html', disaster=disaster, items=items)

	@app.route('/disasters/<int:disaster_id>/requests/new', methods=['GET', 'POST'])
	def create_resource_request(disaster_id: int):
		"""Create a new resource request and auto-assign ML priority."""
		disaster = Disaster.query.get_or_404(disaster_id)

		if request.method == 'POST':
			try:
				severity_level = request.form.get('severity_level', '').strip().lower()
				people_affected = int(request.form.get('people_affected', '0') or 0)
				resource_type = request.form.get('resource_type', '').strip().lower()
				location_urgency = request.form.get('location_urgency', '').strip().lower()

				allowed_severity = {'low', 'medium', 'high'}
				allowed_resource_types = {'food', 'medical', 'shelter'}
				allowed_location_urgency = {'low', 'medium', 'high', ''}

				if severity_level not in allowed_severity:
					flash('Severity level must be low, medium, or high.', 'danger')
					return redirect(url_for('create_resource_request', disaster_id=disaster_id))

				if people_affected <= 0:
					flash('People affected must be a positive number.', 'danger')
					return redirect(url_for('create_resource_request', disaster_id=disaster_id))

				if resource_type not in allowed_resource_types:
					flash('Resource type must be food, medical, or shelter.', 'danger')
					return redirect(url_for('create_resource_request', disaster_id=disaster_id))

				if location_urgency not in allowed_location_urgency:
					flash('Location urgency must be low, medium, high, or empty.', 'danger')
					return redirect(url_for('create_resource_request', disaster_id=disaster_id))

				predicted_priority = resource_priority_service.predict_priority(
					severity_level=severity_level,
					people_affected=people_affected,
					resource_type=resource_type,
					location_urgency=location_urgency or None
				)

				resource_request = ResourceRequest(
					DisasterID=disaster_id,
					SeverityLevel=severity_level,
					PeopleAffected=people_affected,
					ResourceType=resource_type,
					LocationUrgency=location_urgency or None,
					PredictedPriority=predicted_priority
				)

				db.session.add(resource_request)
				db.session.commit()

				flash(
					f'Resource request created with AI-assigned priority: {predicted_priority}.',
					'success'
				)
				return redirect(url_for('disaster_detail', disaster_id=disaster_id))

			except Exception as e:
				db.session.rollback()
				flash(f'Error creating resource request: {str(e)}', 'danger')
				return redirect(url_for('create_resource_request', disaster_id=disaster_id))

		return render_template('resource_request_form.html', disaster=disaster)

	@app.route('/disasters/<int:disaster_id>/tables')
	def show_disaster_tables(disaster_id: int):
		"""Show all tables for a specific disaster (Multi-tenant view)"""
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get all data from disaster-specific tables
		warehouses = TenantWarehouse.query.all()
		items = TenantReliefItem.query.all()
		beneficiaries = TenantBeneficiary.query.all()
		distributions = TenantDistribution.query.all()
		
		return render_template('disaster_tables.html', 
			disaster=disaster,
			warehouses=warehouses,
			items=items,
			beneficiaries=beneficiaries,
			distributions=distributions)

	@app.route('/disasters/<int:disaster_id>/analyze')
	def analyze(disaster_id: int):
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get data for analysis
		warehouses = TenantWarehouse.query.all()
		items = TenantReliefItem.query.all()
		beneficiaries = TenantBeneficiary.query.all()
		distributions = TenantDistribution.query.all()
		
		return render_template('analyze.html', 
			disaster=disaster,
			warehouses=warehouses,
			items=items,
			beneficiaries=beneficiaries,
			distributions=distributions)

	@app.route('/disasters/<int:disaster_id>/predict')
	def predict(disaster_id: int):
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get data for prediction
		warehouses = TenantWarehouse.query.all()
		items = TenantReliefItem.query.all()
		beneficiaries = TenantBeneficiary.query.all()
		distributions = TenantDistribution.query.all()
		
		# Get ML predictions (includes trend analysis)
		ml_data = ml_service.get_prediction_data(disaster_id, warehouses, items, beneficiaries, distributions)
		
		# Get optimization suggestions
		optimization_suggestions = ml_service.get_optimization_suggestions(warehouses, items, distributions)
		
		return render_template('predict.html', 
			disaster=disaster,
			warehouses=warehouses,
			items=items,
			beneficiaries=beneficiaries,
			distributions=distributions,
			ml_data=ml_data,
			trend_analysis=ml_data.get('trend_analysis', {}),
			optimization_suggestions=optimization_suggestions)
	
	@app.route('/disasters/<int:disaster_id>/predict/api')
	def predict_api(disaster_id: int):
		"""API endpoint for ML predictions"""
		disaster = Disaster.query.get_or_404(disaster_id)
		
		# Get tenant-specific models for this disaster
		TenantWarehouse, TenantReliefItem, TenantBeneficiary, TenantDistribution = get_tenant_models(disaster_id)
		
		# Get data for prediction
		warehouses = TenantWarehouse.query.all()
		items = TenantReliefItem.query.all()
		beneficiaries = TenantBeneficiary.query.all()
		distributions = TenantDistribution.query.all()
		
		# Get ML predictions
		ml_data = ml_service.get_prediction_data(disaster_id, warehouses, items, beneficiaries, distributions)
		
		return jsonify(ml_data)

	@app.route('/test-ml')
	def test_ml_endpoint():
		"""
		Basic ML test endpoint using one sample request input.
		Useful for quickly checking model behavior via browser/API.
		"""
		sample_input = {
			"severity_level": "high",
			"people_affected": 250,
			"resource_type": "medical",
			"location_urgency": "high"
		}

		predicted_priority = resource_priority_service.predict_priority(**sample_input)

		return jsonify({
			"status": "success",
			"message": "ML model test completed.",
			"input": sample_input,
			"predicted_priority": predicted_priority
		})


app = create_app()


if __name__ == '__main__':
	app.run(debug=True)


